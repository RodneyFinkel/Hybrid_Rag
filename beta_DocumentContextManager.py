# Using chromadb
import chromadb #import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer # Upgrade from BERT embeddings
import numpy as np
import time
import logging
import uuid
from redis_client import cache_result, redis_client # Import caching decorator and client
import json


# New Imports for Hybrid Search
from rank_bm25 import BM25Okapi
from ragatouille import RAGPretrainedModel  # For ColBERT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentContextManager:
    def __init__(self, similarity_threshold=0.4):
        self.id = str(uuid.uuid4())
        self.client = chromadb.PersistentClient(path="./chroma_storage")  # ← Change to this for persistence
        logging.info("Persistent Chroma Initialized - Data auto-saves to ./chroma_storage/")
        self.collection = self.client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"}) # Ensure cosine similarity is used
        logging.info(f"Chroma collection loaded: {self.collection.count()} total documents persisted")
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # New model
        logging.info("SentenceTransformer Initialized")
        
        # Store similarity threshold
        self.similarity_threshold = similarity_threshold
        logging.info(f"Initialized with similarity threshold: {self.similarity_threshold}")
        
        # CHANGE: Initialize last_raw_results to store raw retrieval data for debugging
        self.last_raw_results = []
        logging.info("initializing self.last_raw_results-pre get similar docs")
        self.retrieval_config = {
            'hybrid_enabled': True,  # Toggle hybrid search
            'semantic_weight': 0.7,   # Weight for semantic score in fusion (0-1)
            'bm25_weight': 0.3,       # Weight for BM25 score in fusion (0-1)
            'bm25_k1': 1.2,           # BM25 term saturation
            'bm25_b': 0.75,           # BM25 length normalization
            'rerank_enabled': True,  # Toggle ColBERT reranking
            'rerank_k': 50,           # Initial retrieve this many for reranking, then take top_k
            'colbert_model': 'colbert-ir/colbertv2.0'  # Pretrained ColBERT model
        }    
        
        # New: Preload ColBERT if enabled (lazy load on first use)
        self.colbert_reranker = None
        # New: BM25 index (built on add document)
        self.bm25_index = None
        self.documents_for_bm25 = [] # list of tokenized docs for BM25
        
        
    # =================SETTERS/GETTERS=================
    
    def set_similarity_threshold(self, threshold):  # NEW: Set the similarity threshold for document retrieval
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            raise ValueError("Similarity threshold must be a number between 0 and 1")
        self.similarity_threshold = float(threshold)
        logging.info(f"Updated similarity threshold to: {self.similarity_threshold}")
        
    # New: Setters and getters for retrieval config
    def set_retrieval_config(self, config):
        self.retrieval_config.update(config)
        logging.info(f"Updated retrieval config: {self.retrieval_config}")
        if self.retrieval_config['rerank_enabled'] and not self.colbert_reranker:
            self.colbert_reranker = RAGPretrainedModel.from_pretrained(self.retrieval_config['colbert_model'])
            logging.info(f"Loaded ColBERT model: {self.retrieval_config['colbert_model']}")
            
    def get_retrieval_config(self):
        return self.retrieval_config
    
        # ================= MULTI-CHUNK / MULTI-HOP HELPERS =================

    def _group_chunks_by_document(self, raw_results):
        """
        raw_results: self.last_raw_results
        Returns {doc_id: [chunk_dicts]}
        """
        grouped = {}
        for r in raw_results:
            grouped.setdefault(r["doc_id"], []).append(r)
        return grouped

    def _score_documents(self, grouped_chunks, lambda_tail=0.25):
        """
        Aggregate chunk scores into document-level scores.
        """
        scored_docs = []
        for doc_id, chunks in grouped_chunks.items():
            chunks = sorted(chunks, key=lambda x: x["similarity"], reverse=True)
            head = chunks[0]["similarity"]
            tail = sum(c["similarity"] for c in chunks[1:]) * lambda_tail
            scored_docs.append({
                "doc_id": doc_id,
                "score": head + tail,
                "chunks": chunks
            })
        return sorted(scored_docs, key=lambda x: x["score"], reverse=True)

    def _is_multihop_query(self, query: str) -> bool:
        triggers = [
            "how", "why", "compare", "difference",
            "relationship", "leads to", "impact"
        ]
        q = query.lower()
        return any(t in q for t in triggers)

        
    
    # NEW: using Sentence Transformer
    def _embed_text(self, text):
        embedding = self.model.encode(text, show_progress_bar=True)
        if isinstance(embedding, np.ndarray):
            #embedding = embedding.tolist()
            embedding = embedding
        elif not isinstance(embedding, list):
            logging.error(f"Unexpected embedding type: {type(embedding)}")
            raise ValueError(f"Unexpected embedding type: {type(embedding)}")
        logging.info(f"Generated Embedding Shape: {len(embedding)}")
        return embedding
    
    # Using chromadb
    def add_document(self, doc_id, text, filename):
        # Check for existing document to avoid duplicates
        try:
            existing = self.collection.get(ids=[doc_id])
            if existing['ids']:
                print(f"Doc {doc_id} already exists. Skipping addition.")
                return
        except:
            pass # Not found, proceed to add
        
        clean_text = " ".join(text.split()) # clean up document text
        embedding = self._embed_text(clean_text)
        metadata = {
            "filename": filename,
            "upload_time": time.time(),
            "summary":text[:50]
        }
        
        logging.info(f"Storing Embedding for Doc ID: {doc_id} with embedding:{embedding[:5]}")
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[clean_text]
        )
        
        # New: Update BM25 index (THIS IS WHERE THE bm25_index is created using the desired context_manager instance)
        tokenized_doc = clean_text.lower().split() # simple tokenization for BM25
        self.documents_for_bm25.append(tokenized_doc)
        logging.info(f"Tokenized document {doc_id}: {tokenized_doc[:10]}...(total {len(tokenized_doc)} tokens)")
        try:
            self.bm25_index = BM25Okapi(self.documents_for_bm25) # rebuild index (efficient for small corpora, optimize for large)
            logging.info(f"Successfully updated BM25 index with {len(self.documents_for_bm25)} documents")
            print(self.bm25_index)
        except Exception as e:
            logging.error(f"Failed to update BM25 index: {str(e)}")
            self.bm25_index = None
            
        
    def normalize_bm25_scores(self, bm25_scores):
        if len(bm25_scores) == 0:
            logging.info("No BM25 scores to normalize (empty list)")
            return bm25_scores
        min_score = min(bm25_scores)
        max_score = max(bm25_scores)
        if max_score == min_score:
            logging.info(f"BM25 scores identical: min={min_score}, max={max_score}, returning zeros")
            return [0.0] * len(bm25_scores)
        normalized_scores = [(score - min_score)/ (max_score - min_score) for score in bm25_scores]
        logging.info(f"BM25 scores normalized: min={min_score}, max={max_score}")
        return normalized_scores
    
                
    # Fix BM25 rebuild call
    def rebuild_bm25_from_chroma(self):
        # Fetch everything we need: text, metadata, and IDs
        data = self.collection.get(include=["documents", "metadatas"])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        ids = data.get("ids", [])
        
        if not documents:
            logging.info("No documents found in Chroma collection for BM25 rebuild")
            self.bm25_index = None
            self.doc_id_to_bm25_index = {}
            return
        
        # Tokenize only non-empty documents
        tokenized_docs = []
        self.doc_id_to_bm25_index = {}  # Map Chroma ID → index in BM25 corpus
        
        for i, doc in enumerate(documents):
            if not doc or not doc.strip():
                continue
            tokenized = doc.lower().split()
            if tokenized:
                tokenized_docs.append(tokenized)
                self.doc_id_to_bm25_index[ids[i]] = len(tokenized_docs) - 1  # Map to position in BM25
        
        if tokenized_docs:
            try:
                self.bm25_index = BM25Okapi(tokenized_docs)
                logging.info(f"Successfully rebuilt BM25 index from {len(tokenized_docs)} persisted documents")
            except Exception as e:
                logging.error(f"Failed to rebuild BM25 index: {str(e)}")
                self.bm25_index = None
                self.doc_id_to_bm25_index = {}
        else:
            logging.info("No valid non-empty documents to build BM25 index")
            self.bm25_index = None
            self.doc_id_to_bm25_index = {}


    
    #  Using Chromadb and Redis Caching______Orchesration of Hybrid Search + Reranking
    
    @cache_result(ttl=600) # Cache results for 10 minutes
    def get_similar_documents(self, query, top_k=10, keyword_filter=None):
        logging.info("initiating get_similar_documents function")
        print(self.bm25_index)
        if len(query.strip()) < 3: # skip very short queries
            print('~Query to short, skipping retrieval.')
            return []
        
        # query_embedding = self._embed_text(query).tolist()
        query_embedding = self._embed_text(query)
        query_params = {
            'query_embeddings': [query_embedding],
            'n_results': top_k if not self.retrieval_config['rerank_enabled'] else self.retrieval_config['rerank_k'],
            'include': ["documents", "metadatas", "distances"]
        }
        
        results = self.collection.query(**query_params)
        
        # Extract inner lists (Chroma returns nested lists for multi-query, but we have one query)
        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else [] # NEW
        
        # NEW: store raw results for debugging and ui
        self.last_raw_results = [
            {
                "doc_id": ids[i],
                "distance": distances[i],
                "similarity": 1 - distances[i],
                "filename": metadatas[i].get("filename", "Unknown") if metadatas else "Unknown",

                # UI/debug only
                "snippet": documents[i][:100] + "..." if documents and len(documents[i]) > 100 else documents[i],

                # REQUIRED for multi-hop + ColBERT
                "full_text": documents[i],

                "bm25_score": 0.0
            } for i in range(len(ids))
        ]

        logging.info("initializing self.last_raw_results in get_similar_documents")
        
                # ================= DOCUMENT-LEVEL AGGREGATION =================
                
        grouped = self._group_chunks_by_document(self.last_raw_results)
        scored_documents = self._score_documents(grouped)

        # Take top-N documents, not chunks
        top_doc_k = 10
        top_documents = scored_documents[:top_doc_k]

        logging.info(f"Selected {len(top_documents)} documents after aggregation")
        
                # ================= MULTI-CHUNK EXPANSION =================
                
        expanded_chunks = []
        window = 2 if self._is_multihop_query(query) else 1

        for doc in top_documents:
            chunks = sorted(doc["chunks"], key=lambda x: x["distance"])
            expanded_chunks.extend(chunks[: window + 1])

        logging.info(f"Expanded to {len(expanded_chunks)} chunks after expansion")

        # ================= USE EXPANDED CHUNKS AS WORKING SET =================
        
        ids = [c["doc_id"] for c in expanded_chunks]
        documents = [c["full_text"] for c in expanded_chunks]
        metadatas = [{"filename": c["filename"]} for c in expanded_chunks]
        distances = [c["distance"] for c in expanded_chunks]

        
        # Search to see if singleton in LanguageModelProcessor uses correct context_manager instance/object
        print(self.retrieval_config)
        print(self.bm25_index)
        logging.info(f"Documents for BM25: {len(self.documents_for_bm25)}")
        logging.debug(f"BM25 index type: {type(self.bm25_index)}")
        
        # New Hybrid Search
        if self.retrieval_config['hybrid_enabled'] and self.bm25_index:
            logging.info("Starting hybrid retrieval")
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            logging.info(f"Raw BM25 scores: {bm25_scores[:5]}... (total {len(bm25_scores)})")
            normalized_bm25_scores = self.normalize_bm25_scores(bm25_scores)
            hybrid_scores = {}
            for i, doc_id in enumerate(ids):
                semantic_sim = 1 - distances[i]
                bm25_idx = self.doc_id_to_bm25_index.get(doc_id)
                bm25_score = (
                    normalized_bm25_scores[bm25_idx]
                    if bm25_idx is not None and bm25_idx < len(normalized_bm25_scores)
                    else 0.0
                )
                # bm25_score = normalized_bm25_scores[i] if i < len(normalized_bm25_scores) else 0 # Align with retrieved docs
                # self.last_raw_results[i]["bm25_score"] = bm25_score # Store normalized BM25 score
                for r in self.last_raw_results:
                    if r["doc_id"] == doc_id:
                        r["bm25_score"] = bm25_score
                        break
                fused_score = (
                    self.retrieval_config['semantic_weight'] * semantic_sim +
                    self.retrieval_config['bm25_weight'] * bm25_score
                )
                hybrid_scores[doc_id] = fused_score
                logging.info(f"Doc {doc_id}: semantic={semantic_sim:.4f}, bm25={bm25_score:.4f}, fused={fused_score:.4f}")


            # Sort by fused score
            sorted_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_k]

            ids = sorted_ids
            documents = [documents[ids.index(i)] for i in sorted_ids]
            metadatas = [metadatas[ids.index(i)] for i in sorted_ids]
            distances = [1.0 - hybrid_scores[i] for i in sorted_ids]
        else:
            logging.info("Hybrid search skipped: hybrid_enabled=%s, bm25_index=%s",
                         self.retrieval_config['hybrid_enabled'], self.bm25_index is not None)        
           
                
       
        # ================= COLBERT RERANK (ON EXPANDED CHUNKS) =================
        
        if self.retrieval_config['rerank_enabled'] and self.colbert_reranker and documents:
            logging.info("Starting ColBERT reranking on expanded chunks")

            # texts = documents[: self.retrieval_config['rerank_k']]
            texts = [
                {"idx": i, "text": documents[i]}
                for i in range(min(len(documents), self.retrieval_config['rerank_k']))
            ]   

            reranked = self.colbert_reranker.rerank(
                query,
                # texts,
                [t['text'] for t in texts],
                k=min(top_k, len(texts))
            )

            scores = [r["score"] for r in reranked]
            min_s, max_s = min(scores), max(scores)

            norm_scores = (
                [(s - min_s) / (max_s - min_s) for s in scores]
                if max_s > min_s else
                [0.0] * len(scores)
            )

            new_ids, new_docs, new_metas, new_dists = [], [], [], []

            for r, ns in zip(reranked, norm_scores):
                idx = next(t["idx"] for t in texts if t["text"] == r["content"]) # Find original index
                new_ids.append(ids[idx])
                new_docs.append(documents[idx])
                new_metas.append(metadatas[idx])
                new_dists.append(1.0 - ns)

            ids, documents, metadatas, distances = (
                new_ids, new_docs, new_metas, new_dists
            )
         
        
        # NEW: Similarity threshold filtering
        similar_docs = []
        for i in range(len(ids)):
            similarity = max(0.0, min(1.0, 1 - distances[i])) if distances else 0  # Clamp to [0,1]
            logging.info(f"Raw distance for {ids[i]}: {distances[i]}, similarity: {similarity}") # Log raw scores
            if similarity >= self.similarity_threshold:
                similar_docs.append({
                    "doc_id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "similarity": similarity # Include similirity score if available
                })
            else:
                logging.debug(f"Document {ids[i]} filtered out(similarity: {similarity} < {self.similarity_threshold})")
                
        
        logging.info(f"Retrieved {len(similar_docs)} documents with similarity >= {self.similarity_threshold}")
        
        # Publish to Redis channela
        publish_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "results": similar_docs  # Your list of dicts (doc_id, document, metadata, similarity)
        }
        redis_client.publish("retrieval_channel", json.dumps(publish_data))
        logging.info("Published retrieval results to Redis pub/sub channel")
        
        return similar_docs

