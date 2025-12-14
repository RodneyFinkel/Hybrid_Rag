function exportHistory(format) {
            // Open in new tab to trigger download
            window.open(`/export_history_${format}`, '_blank');
            console.log(`Exporting history as ${format.toUpperCase()}`);
    }

        // Create tooltip element
        const tooltip = d3.select('body')
            .append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);

        document.addEventListener('DOMContentLoaded', function() {
            // Theme toggle
            const themeToggle = document.getElementById('themeToggle');
            const lightIcon = document.getElementById('lightIcon');
            const darkIcon = document.getElementById('darkIcon');
            themeToggle.addEventListener('click', () => {
                document.documentElement.classList.toggle('dark');
                lightIcon.classList.toggle('hidden');
                darkIcon.classList.toggle('hidden');
            });

            // Fetch and display last retrieval results
        async function fetchLastRetrieval() {
            try {
                const response = await fetch('/get_last_retrieval');
                const data = await response.json();
                const retrievalList = document.getElementById('last-retrieval-list');
                retrievalList.innerHTML = ''; // Clear existing rows
                if (data.results.length === 0) {
                    retrievalList.innerHTML = '<tr><td colspan="3" class="p-2 text-center">No retrieval data available</td></tr>';
                    return;
                }
                data.results.forEach(result => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td class="p-2">${result.filename}</td>
                        <td class="p-2">${result.snippet}</td>
                        <td class="p-2">${result.similarity}%</td>
                    `;
                    retrievalList.appendChild(row);
                });
            } catch (error) {
                console.error('Error fetching last retrieval:', error);
                document.getElementById('last-retrieval-list').innerHTML = '<tr><td colspan="3" class="p-2 text-center">Error loading data</td></tr>';
            }
        }

        document.getElementById('onlineResearchToggle').addEventListener('change', (event) => {
            fetch('/set_online_research', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: event.target.checked})
            })
            .then(response => response.json())
            .then(data => console.log('Online research:', data))
            .catch(error => console.error('Error:', error));
        });

            // Clock
            function updateClock() {
                const now = new Date();
                document.getElementById('clock').textContent = now.toLocaleTimeString();
            }
            setInterval(updateClock, 1000);
            updateClock();

            // Weather
            function fetchWeather() {
                fetch('/weather?city=Haifa')
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            document.getElementById('weatherDisplay').textContent = 'Weather: Error';
                        } else {
                            document.getElementById('weatherDisplay').textContent = `Weather in ${data.city}: ${data.temperature}°C, ${data.description}`;
                        }
                    });
            }
            fetchWeather();

            // Stock Ticker
            function fetchStocks() {
                fetch('/stocks')
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            document.getElementById('stockTicker').textContent = 'Stocks: Error';
                        } else {
                            const tickerText = data.map(stock => `${stock.symbol}: $${stock.price || 'N/A'}`).join(' | ');
                            document.getElementById('stockTicker').textContent = tickerText;
                        }
                    });
            }
            fetchStocks();
            setInterval(fetchStocks, 60000);

            // Quote
            function newQuote() {
                fetch('/quote')
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            document.getElementById('quoteDisplay').textContent = 'Failed to fetch quote';
                        } else {
                            document.getElementById('quoteDisplay').textContent = `"${data.quote}" - ${data.author}`;
                        }
                    });
            }
            newQuote();

            // Transcription
            let pollingInterval;
            function startTranscription() {
                fetch('/start_transcription', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'Transcription started') {
                            document.getElementById('transcriberStatus').textContent = 'Active';
                            document.getElementById('record').classList.add('hidden');
                            startPolling();
                        } else {
                            Toastify({
                                text: data.status,
                                duration: 3000,
                                style: { background: '#ef4444' }
                            }).showToast();
                        }
                    })
                    .catch(error => {
                        Toastify({
                            text: 'Error starting transcription: ' + error,
                            duration: 3000,
                            style: { background: '#ef4444' }
                        }).showToast();
                    });
            }

            function stopTranscription() {
                fetch('/stop_transcription', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('transcriberStatus').textContent = 'Idle';
                        document.getElementById('startIcon')?.classList.remove('hidden');
                        stopPolling();
                        Toastify({
                            text: data.status,
                            duration: 3000,
                            style: { background: '#10b981' }
                        }).showToast();
                    })
                    .catch(error => {
                        Toastify({
                            text: 'Error stopping transcription: ' + error,
                            duration: 3000,
                            style: { background: '#ef4444' }
                        }).showToast();
                    });
            }

            function startPolling() {
                pollingInterval = setInterval(() => {
                    fetch('/get_data')
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'Active') {
                                document.getElementById('transcript').textContent = data.transcript || 'No transcript available yet.';
                                document.getElementById('response').textContent = data.llm_response || 'No response yet.';
                            } else {
                                document.getElementById('transcript').textContent = 'Transcription inactive.';
                                document.getElementById('response').textContent = 'No response yet.';
                            }
                        });
                }, 2000);
            }

            function stopPolling() {
                clearInterval(pollingInterval);
            }


            // NEW: Submit Typed Query
            function submitQuery() {
                const query = document.getElementById('manualQuery').value.trim();
                if (!query) {
                    Toastify({
                        text: 'Please enter a query.',
                        duration: 3000,
                        style: { background: '#ef4444' }
                    }).showToast();
                    return;
                }
                document.getElementById('transcript').textContent = `Query: ${query}`;
                document.getElementById('response').textContent = 'Processing...';
                fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            Toastify({
                                text: data.error,
                                duration: 3000,
                                style: { background: '#ef4444' }
                            }).showToast();
                            document.getElementById('response').textContent = 'Error processing query.';
                        } else {
                            document.getElementById('response').textContent = data.response || 'No response yet.';
                            updateRetrievalTable();  // Refresh table after query
                            Toastify({
                                text: 'Query processed successfully!',
                                duration: 3000,
                                style: { background: '#10b981' }
                            }).showToast();
                            
                        }
                    })
                    .catch(error => {
                        Toastify({
                            text: 'Error submitting query: ' + error,
                            duration: 3000,
                            style: { background: '#ef4444' }
                        }).showToast();
                        document.getElementById('response').textContent = 'Error processing query.';
                    });
            }

            // PDF Upload
            function uploadPDF() {
                const pdfInput = document.getElementById('pdfUpload');
                const progressBar = document.getElementById('progressBar');
                const uploadProgress = document.getElementById('uploadProgress');
                if (pdfInput.files.length === 0) {
                    Toastify({
                        text: 'Please select at least one PDF file.',
                        duration: 3000,
                        style: { background: '#ef4444' }
                    }).showToast();
                    return;
                }
                const formData = new FormData();
                for (let file of pdfInput.files) {
                    formData.append('pdf', file);
                }
                uploadProgress.classList.remove('hidden');
                progressBar.style.width = '0%';
                fetch('/upload_pdf', {
                    method: 'POST',
                    body: formData
                })
                    .then(response => response.json())
                    .then(data => {
                        progressBar.style.width = '100%';
                        setTimeout(() => uploadProgress.classList.add('hidden'), 500);
                        Toastify({
                            text: data.status,
                            duration: 3000,
                            style: { background: '#10b981' }
                        }).showToast();
                        listDocuments();
                    })
                    .catch(error => {
                        progressBar.style.width = '100%';
                        setTimeout(() => uploadProgress.classList.add('hidden'), 500);
                        Toastify({
                            text: 'Error uploading PDF: ' + error,
                            duration: 3000,
                            style: { background: '#ef4444' }
                        }).showToast();
                    });
            }

            // Documents
            let allDocuments = [];
            function listDocuments() {
                fetch('/get_documents')
                    .then(response => response.json())
                    .then(documents => {
                        allDocuments = documents;
                        const searchQuery = document.getElementById('documentSearch').value.toLowerCase();
                        const filteredDocuments = searchQuery
                            ? documents.filter(doc =>
                                doc.filename.toLowerCase().includes(searchQuery) ||
                                doc.summary.toLowerCase().includes(searchQuery)
                            )
                            : documents;
                        const tbody = document.getElementById('documentsBody');
                        tbody.innerHTML = '';
                        filteredDocuments.forEach(doc => {
                            const row = document.createElement('tr');
                            row.classList.add('animate-fadeIn');
                            row.innerHTML = `
                                <td class="py-2 text-left text-gray-700 dark:text-gray-200">${doc.filename}</td>
                                <td class="py-2 text-left text-gray-700 dark:text-gray-200">${doc.summary}</td>
                                <td class="py-2 text-left text-gray-700 dark:text-gray-200">${doc.upload_time}</td>
                                <td class="py-2 text-left">
                                    <button onclick="deleteDocument('${doc.doc_id}')" class="group relative glass-button border border-gray-200/20 px-2 py-1 rounded-full hover:bg-red-100/30 dark:hover:bg-red-900/30 text-red-600 hover:text-red-700 text-sm font-medium transition-all duration-200">
                                        Delete
                                        <span class="tooltip absolute bg-gray-800 text-white text-xs rounded py-1 px-2 -top-10 left-1/2 transform -translate-x-1/2">Delete document</span>
                                    </button>
                                </td>
                            `;
                            tbody.appendChild(row);
                        });
                    });
            }
            listDocuments();

            // Delete Document
            function deleteDocument(docId) {
                if (confirm(`Are you sure you want to delete document ${docId}?`)) {
                    fetch('/delete_document', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ doc_id: docId })
                    })
                        .then(response => response.json())
                        .then(data => {
                            Toastify({
                                text: data.status,
                                duration: 3000,
                                style: { background: '#10b981' }
                            }).showToast();
                            listDocuments();
                        })
                        .catch(error => {
                            Toastify({
                                text: 'Error deleting document: ' + error,
                                duration: 3000,
                                style: { background: '#ef4444' }
                            }).showToast();
                        });
                }
            }

            // Search Documents
            document.getElementById('documentSearch').addEventListener('input', listDocuments);

            // Config Accordion
            function toggleAccordion(id) {
                const element = document.getElementById(id);
                const arrow = document.getElementById(id === 'searchConfig' ? 'searchArrow' : 'chunkingArrow');
                element.classList.toggle('hidden');
                arrow.classList.toggle('rotate-180');
            }

            // Search Config
            const similaritySlider = document.getElementById('similarity_threshold');
            const similarityValue = document.getElementById('similarity_value');
            similaritySlider.addEventListener('input', () => {
                similarityValue.textContent = similaritySlider.value;
            });

            const semanticWeightSlider = document.getElementById('semantic_weight');
            const semanticWeightValue = document.getElementById('semantic_weight_value');
            const bm25WeightSlider = document.getElementById('bm25_weight');
            const bm25WeightValue = document.getElementById('bm25_weight_value');

            // Synchronize weight sliders to sum to 1
            semanticWeightSlider.addEventListener('input', () => {
                semanticWeightValue.textContent = semanticWeightSlider.value;
                bm25WeightSlider.value = (1 - parseFloat(semanticWeightSlider.value)).toFixed(2);
                bm25WeightValue.textContent = bm25WeightSlider.value;
            });
            bm25WeightSlider.addEventListener('input', () => {
                bm25WeightValue.textContent = bm25WeightSlider.value;
                semanticWeightSlider.value = (1 - parseFloat(bm25WeightSlider.value)).toFixed(2);
                semanticWeightValue.textContent = semanticWeightSlider.value;
            });

            const bm25K1Slider = document.getElementById('bm25_k1');
            const bm25K1Value = document.getElementById('bm25_k1_value');
            bm25K1Slider.addEventListener('input', () => {
                bm25K1Value.textContent = bm25K1Slider.value;
            });

            const bm25BSlider = document.getElementById('bm25_b');
            const bm25BValue = document.getElementById('bm25_b_value');
            bm25BSlider.addEventListener('input', () => {
                bm25BValue.textContent = bm25BSlider.value;
            });

            function updateRetrievalConfig() {
                const updateButton = document.querySelector('#searchConfig button[onclick="updateRetrievalConfig()"]');
                const spinner = document.getElementById('updateSpinner');
                updateButton.disabled = true;
                spinner.style.display = 'inline-block';

                const data = {
                    hybrid_enabled: document.getElementById('hybrid_enabled').checked,
                    semantic_weight: parseFloat(document.getElementById('semantic_weight').value),
                    bm25_weight: parseFloat(document.getElementById('bm25_weight').value),
                    bm25_k1: parseFloat(document.getElementById('bm25_k1').value),
                    bm25_b: parseFloat(document.getElementById('bm25_b').value),
                    rerank_enabled: document.getElementById('rerank_enabled').checked,
                    rerank_k: parseInt(document.getElementById('rerank_k').value),
                    colbert_model: document.getElementById('colbert_model').value
                };

                // Client-side validation
                if (data.hybrid_enabled && Math.abs(data.semantic_weight + data.bm25_weight - 1.0) > 0.01) {
                    Toastify({
                        text: 'Semantic and BM25 weights must sum to 1',
                        duration: 3000,
                        style: { background: '#ef4444' }
                    }).showToast();
                    updateButton.disabled = false;
                    spinner.style.display = 'none';
                    return;
                }
                if (data.rerank_k < 1) {
                    Toastify({
                        text: 'Rerank Top K must be at least 1',
                        duration: 3000,
                        style: { background: '#ef4444' }
                    }).showToast();
                    updateButton.disabled = false;
                    spinner.style.display = 'none';
                    return;
                }

                console.log('Sending retrieval config:', data);

                fetch('/set_retrieval_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                })
                    .then(response => {
                        console.log('Response status:', response.status);
                        if (!response.ok) {
                            return response.text().then(text => { throw new Error(`HTTP ${response.status}: ${text}`); });
                        }
                        return response.json();
                    })
                    .then(result => {
                        console.log('Response data:', result);
                        Toastify({
                            text: result.status,
                            duration: 3000,
                            style: { background: '#10b981' }
                        }).showToast();
                        fetchRetrievalConfig();
                        updateButton.disabled = false;
                        spinner.style.display = 'none';
                    })
                    .catch(error => {
                        console.error('Error updating retrieval config:', error);
                        Toastify({
                            text: 'Error updating retrieval config: ' + error.message,
                            duration: 3000,
                            style: { background: '#ef4444' }
                        }).showToast();
                        updateButton.disabled = false;
                        spinner.style.display = 'none';
                    });
            }

            function resetRetrievalConfig() {
                const defaultConfig = {
                    hybrid_enabled: false,
                    semantic_weight: 0.7,
                    bm25_weight: 0.3,
                    bm25_k1: 1.2,
                    bm25_b: 0.75,
                    rerank_enabled: false,
                    rerank_k: 50,
                    colbert_model: 'colbert-ir/colbertv2.0'
                };
                document.getElementById('hybrid_enabled').checked = defaultConfig.hybrid_enabled;
                document.getElementById('semantic_weight').value = defaultConfig.semantic_weight;
                document.getElementById('semantic_weight_value').textContent = defaultConfig.semantic_weight;
                document.getElementById('bm25_weight').value = defaultConfig.bm25_weight;
                document.getElementById('bm25_weight_value').textContent = defaultConfig.bm25_weight;
                document.getElementById('bm25_k1').value = defaultConfig.bm25_k1;
                document.getElementById('bm25_k1_value').textContent = defaultConfig.bm25_k1;
                document.getElementById('bm25_b').value = defaultConfig.bm25_b;
                document.getElementById('bm25_b_value').textContent = defaultConfig.bm25_b;
                document.getElementById('rerank_enabled').checked = defaultConfig.rerank_enabled;
                document.getElementById('rerank_k').value = defaultConfig.rerank_k;
                document.getElementById('colbert_model').value = defaultConfig.colbert_model;
                updateRetrievalConfig();
            }

            function setPreset(preset) {
                const presets = {
                    semantic: {
                        hybrid_enabled: false,
                        semantic_weight: 1.0,
                        bm25_weight: 0.0,
                        bm25_k1: 1.2,
                        bm25_b: 0.75,
                        rerank_enabled: false,
                        rerank_k: 50,
                        colbert_model: 'colbert-ir/colbertv2.0'
                    },
                    balanced: {
                        hybrid_enabled: true,
                        semantic_weight: 0.5,
                        bm25_weight: 0.5,
                        bm25_k1: 1.2,
                        bm25_b: 0.75,
                        rerank_enabled: true,
                        rerank_k: 50,
                        colbert_model: 'colbert-ir/colbertv2.0'
                    },
                    keyword: {
                        hybrid_enabled: true,
                        semantic_weight: 0.2,
                        bm25_weight: 0.8,
                        bm25_k1: 1.5,
                        bm25_b: 0.9,
                        rerank_enabled: true,
                        rerank_k: 50,
                        colbert_model: 'colbert-ir/colbertv2.0'
                    }
                };
                const config = presets[preset];
                document.getElementById('hybrid_enabled').checked = config.hybrid_enabled;
                document.getElementById('semantic_weight').value = config.semantic_weight;
                document.getElementById('semantic_weight_value').textContent = config.semantic_weight;
                document.getElementById('bm25_weight').value = config.bm25_weight;
                document.getElementById('bm25_weight_value').textContent = config.bm25_weight;
                document.getElementById('bm25_k1').value = config.bm25_k1;
                document.getElementById('bm25_k1_value').textContent = config.bm25_k1;
                document.getElementById('bm25_b').value = config.bm25_b;
                document.getElementById('bm25_b_value').textContent = config.bm25_b;
                document.getElementById('rerank_enabled').checked = config.rerank_enabled;
                document.getElementById('rerank_k').value = config.rerank_k;
                document.getElementById('colbert_model').value = config.colbert_model;
                updateRetrievalConfig();
            }

            function updateConfigStatus(data) {
                const status = document.getElementById('configStatus');
                status.textContent = data.hybrid_enabled 
                    ? `Hybrid: On, Rerank: ${data.rerank_enabled ? 'On' : 'Off'}`
                    : `Hybrid: Off, Rerank: ${data.rerank_enabled ? 'On' : 'Off'}`;
                status.className = `text-xs px-2 py-1 rounded-full ${
                    data.hybrid_enabled || data.rerank_enabled 
                        ? 'bg-green-200/80 dark:bg-green-700/80 text-green-800 dark:text-green-200' 
                        : 'bg-gray-200/80 dark:bg-gray-700/80 text-gray-600 dark:text-gray-300'
                }`;
            }

            function fetchRetrievalConfig() {
                fetch('/get_retrieval_config')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('hybrid_enabled').checked = data.hybrid_enabled;
                        document.getElementById('semantic_weight').value = data.semantic_weight;
                        document.getElementById('semantic_weight_value').textContent = data.semantic_weight;
                        document.getElementById('bm25_weight').value = data.bm25_weight;
                        document.getElementById('bm25_weight_value').textContent = data.bm25_weight;
                        document.getElementById('bm25_k1').value = data.bm25_k1;
                        document.getElementById('bm25_k1_value').textContent = data.bm25_k1;
                        document.getElementById('bm25_b').value = data.bm25_b;
                        document.getElementById('bm25_b_value').textContent = data.bm25_b;
                        document.getElementById('rerank_enabled').checked = data.rerank_enabled;
                        document.getElementById('rerank_k').value = data.rerank_k;
                        document.getElementById('colbert_model').value = data.colbert_model;
                        document.getElementById('similarity_threshold').value = data.similarity_threshold;
                        document.getElementById('similarity_value').textContent = data.similarity_threshold;
                        updateConfigStatus(data);
                    })
                    .catch(error => {
                        Toastify({
                            text: 'Error fetching retrieval config: ' + error,
                            duration: 3000,
                            style: { background: '#ef4444' }
                        }).showToast();
                    });
            }
            fetchRetrievalConfig();

            // Chunking Config
            function fetchChunkingConfig() {
                fetch('/get_chunking_config')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('ingest_chunk_size').value = data.chunk_size_ingest;
                        document.getElementById('ingest_overlap').value = data.chunk_overlap_ingest;
                        document.getElementById('llm_chunk_size').value = data.chunk_size_llm;
                        document.getElementById('llm_overlap').value = data.chunk_overlap_llm;
                        document.getElementById('similarity_threshold').value = data.similarity_threshold;
                        document.getElementById('similarity_value').textContent = data.similarity_threshold;
                        // NEW: Set chunking_type and semantic_threshold from server
                        document.getElementById('chunking_type').value = data.chunking_type || 'semantic';
                        document.getElementById('semantic_threshold').value = data.semantic_threshold || 0.6;
                        document.getElementById('semantic_threshold_value').textContent = data.semantic_threshold || 0.6;// NEW: Set chunking_type and semantic_threshold from server
                        document.getElementById('chunking_type').value = data.chunking_type || 'semantic';
                        document.getElementById('semantic_threshold').value = data.semantic_threshold || 0.6;
                        document.getElementById('semantic_threshold_value').textContent = data.semantic_threshold || 0.6;
                    })
                    .catch(error => {
                        Toastify({
                            text: 'Error fetching chunking config: ' + error,
                            duration: 3000,
                            style: { background: '#ef4444' }
                        }).showToast();
                    });
            }
            fetchChunkingConfig();


            function updateChunkingConfig() {
                const data = {
                    chunk_size_ingest: document.getElementById('ingest_chunk_size').value,
                    chunk_overlap_ingest: document.getElementById('ingest_overlap').value,
                    chunk_size_llm: document.getElementById('llm_chunk_size').value,
                    chunk_overlap_llm: document.getElementById('llm_overlap').value,
                    similarity_threshold: document.getElementById('similarity_threshold').value,
                    // NEW: Add chunking_type and semantic_threshold to the data object
                    chunking_type: document.getElementById('chunking_type').value,
                    semantic_threshold: parseFloat(document.getElementById('semantic_threshold').value)
                };
                fetch('/set_chunking_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                })
                    .then(response => response.json())
                    .then(result => {
                        Toastify({
                            text: result.status,
                            duration: 3000,
                            style: { background: '#10b981' }
                        }).showToast();
                    })
                    .catch(error => {
                        Toastify({
                            text: 'Error updating chunking config: ' + error,
                            duration: 3000,
                            style: { background: '#ef4444' }
                        }).showToast();
                    });
            }

            // NEW: Event listener for semantic_threshold slider display
            document.getElementById('semantic_threshold').addEventListener('input', (e) => {
                document.getElementById('semantic_threshold_value').textContent = e.target.value;
            });

            function resetChunkingConfig() {
                const defaultConfig = {
                    chunk_size_ingest: 1000,
                    chunk_overlap_ingest: 100,
                    chunk_size_llm: 1000,
                    chunk_overlap_llm: 100,
                    similarity_threshold: 0.3
                };
                document.getElementById('ingest_chunk_size').value = defaultConfig.chunk_size_ingest;
                document.getElementById('ingest_overlap').value = defaultConfig.chunk_overlap_ingest;
                document.getElementById('llm_chunk_size').value = defaultConfig.chunk_size_llm;
                document.getElementById('llm_overlap').value = defaultConfig.chunk_overlap_llm;
                document.getElementById('similarity_threshold').value = defaultConfig.similarity_threshold;
                document.getElementById('similarity_value').textContent = defaultConfig.similarity_threshold;
                // NEW: Reset chunking_type and semantic_threshold
                document.getElementById('chunking_type').value = defaultConfig.chunking_type;
                document.getElementById('semantic_threshold').value = defaultConfig.semantic_threshold;
                document.getElementById('semantic_threshold_value').textContent = defaultConfig.semantic_threshold;
                updateChunkingConfig();
            }

            // Update Retrieval Table
            function updateRetrievalTable() {
                fetch('/get_last_retrieval')
                    .then(response => {
                        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                        return response.json();
                    })
                    .then(data => {
                        const tbody = document.querySelector('#retrievalTable tbody');
                        tbody.innerHTML = '';
                        if (data.status === 'Success' && data.results && data.results.length > 0) {
                            data.results.forEach(result => {
                                const row = document.createElement('tr');
                                row.innerHTML = `
                                    <td>${result.filename || 'Unknown'}</td>
                                    <td>${result.snippet || 'No snippet'}</td>
                                    <td>${(result.similarity * 100).toFixed(1)}%</td>
                                `;
                                tbody.appendChild(row);
                            });
                            console.log(`Added ${data.results.length} rows to retrieval table`);
                        } else {
                            tbody.innerHTML = '<tr><td colspan="3">No retrieval data available</td></tr>';
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching retrieval:', error);
                        const tbody = document.querySelector('#retrievalTable tbody');
                        tbody.innerHTML = '<tr><td colspan="3">Error loading results</td></tr>';
                    });
            }


            // Render Bar Chart for Similarity and BM25 Scores
            function renderScoreChart() {
                fetch('/get_last_retrieval')
                    .then(response => {
                        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                        return response.json();
                    })
                    .then(data => {
                        if (data.status !== 'Success' || !data.results || data.results.length === 0) {
                            console.log('No data for score chart');
                            d3.select('#scoreChart').selectAll('*').remove();
                            d3.select('#scoreChart').append('p').text('No data available for chart');
                            return;
                        }

                        const margin = { top: 20, right: 30, bottom: 100, left: 60 };
                        const width = 800 - margin.left - margin.right;
                        const height = 400 - margin.top - margin.bottom;

                        d3.select('#scoreChart').selectAll('*').remove();
                        const svg = d3.select('#scoreChart')
                            .append('svg')
                            .attr('width', width + margin.left + margin.right)
                            .attr('height', height + margin.top + margin.bottom)
                            .append('g')
                            .attr('transform', `translate(${margin.left},${margin.top})`);

                        const x = d3.scaleBand()
                            .domain(data.results.map(d => d.doc_id))
                            .range([0, width])
                            .padding(0.4);

                        const y = d3.scaleLinear()
                            .domain([0, 1])
                            .range([height, 0]);

                        // Similarity bars
                        svg.selectAll('.similarity-bar')
                            .data(data.results)
                            .enter()
                            .append('rect')
                            .attr('class', 'similarity-bar')
                            .attr('x', d => x(d.doc_id))
                            .attr('y', d => y(d.similarity))
                            .attr('width', x.bandwidth() / 2)
                            .attr('height', d => height - y(d.similarity))
                            .attr('fill', '#4CAF50')
                            .on('mouseover', function(event, d) {
                                d3.select(this).attr('opacity', 0.8);
                                tooltip.style('opacity', 1)
                                    .html(`<strong>${d.filename}</strong><br>Similarity: ${(d.similarity * 100).toFixed(1)}%`)
                                    .style('left', (event.pageX + 10) + 'px')
                                    .style('top', (event.pageY - 10) + 'px');
                            })
                            .on('mouseout', function() {
                                d3.select(this).attr('opacity', 1);
                                tooltip.style('opacity', 0);
                            });

                        // BM25 bars
                        svg.selectAll('.bm25-bar')
                            .data(data.results)
                            .enter()
                            .append('rect')
                            .attr('class', 'bm25-bar')
                            .attr('x', d => x(d.doc_id) + x.bandwidth() / 2)
                            .attr('y', d => y(d.bm25_score || 0))
                            .attr('width', x.bandwidth() / 2)
                            .attr('height', d => height - y(d.bm25_score || 0))
                            .attr('fill', '#2196F3')
                            .on('mouseover', function(event, d) {
                                d3.select(this).attr('opacity', 0.8);
                                tooltip.style('opacity', 1)
                                    .html(`<strong>${d.filename}</strong><br>BM25: ${(d.bm25_score || 0).toFixed(2)}`)
                                    .style('left', (event.pageX + 10) + 'px')
                                    .style('top', (event.pageY - 10) + 'px');
                            })
                            .on('mouseout', function() {
                                d3.select(this).attr('opacity', 1);
                                tooltip.style('opacity', 0);
                            });

                        // Axes
                        svg.append('g')
                            .attr('transform', `translate(0,${height})`)
                            .call(d3.axisBottom(x))
                            .selectAll('text')
                            .attr('transform', 'rotate(-45)')
                            .style('text-anchor', 'end');

                        svg.append('g')
                            .call(d3.axisLeft(y).tickFormat(d => `${(d * 100).toFixed(0)}%`));

                        // Labels
                        svg.append('text')
                            .attr('x', width / 2)
                            .attr('y', -10)
                            .attr('text-anchor', 'middle')
                            .text('Similarity and BM25 Scores');

                        svg.append('text')
                            .attr('x', -height / 2)
                            .attr('y', -40)
                            .attr('transform', 'rotate(-90)')
                            .attr('text-anchor', 'middle')
                            .text('Score');
                    })
                    .catch(error => console.error('Error rendering score chart:', error));
            }

            // Update Chunking Info
            function updateChunkingInfo() {
                fetch('/get_chunking_info')
                    .then(response => {
                        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                        return response.json();
                    })
                    .then(data => {
                        const chunkingElem = document.getElementById('chunkingInfo');
                        chunkingElem.textContent = JSON.stringify(data, null, 2);
                    })
                    .catch(error => console.error('Error fetching chunking info:', error));
        }

            // Refresh Info (polling for both table and chunking)
            // function refreshInfo() {
            //     updateRetrievalTable();
            //     updateChunkingInfo();
            //     renderScoreChart();  // Update chart
            // }
            // setInterval(refreshInfo, 5000);
            // refreshInfo();


            // REPLACE THE ENTIRE BLOCK ABOVE WITH THIS (removes polling, adds SSE)
            function refreshInfo(data) {
                updateRetrievalTable(data.results);  // Pass new results
                updateChunkingInfo();               // Keep if needed, or make real-time too
                renderScoreChart();                 // Your existing chart function
            }

            // SSE listener for real-time updates
            const evtSource = new EventSource('/stream_retrieval');
            evtSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                console.log("Real-time retrieval via Redis Pub/Sub:", data);
                refreshInfo(data);  // Update with pushed data
            };

            evtSource.onerror = function() {
                console.error("SSE error — falling back to polling");
                evtSource.close();
                // Fallback: restart polling if SSE fails
                setInterval(() => {
                    fetch('/get_last_retrieval').then(res => res.json()).then(data => refreshInfo(data));
                }, 10000);
            };

            

            // Sidebar Toggle
            const sidebarToggle = document.getElementById('sidebarToggle');
            const sidebar = document.getElementById('sidebar');
            sidebarToggle.addEventListener('click', () => {
                sidebar.classList.toggle('hidden');
            });

            // Expose functions to global scope
            window.startTranscription = startTranscription;
            window.stopTranscription = stopTranscription;
            window.uploadPDF = uploadPDF;
            window.listDocuments = listDocuments;
            window.deleteDocument = deleteDocument;
            window.newQuote = newQuote;
            window.toggleAccordion = toggleAccordion;
            window.updateRetrievalConfig = updateRetrievalConfig;
            window.resetRetrievalConfig = resetRetrievalConfig;
            window.setPreset = setPreset;
            window.updateChunkingConfig = updateChunkingConfig;
            window.resetChunkingConfig = resetChunkingConfig;
            window.submitQuery = submitQuery;
        });
    