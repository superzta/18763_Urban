import React, { useState, useEffect, useRef } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { Camera, Smartphone, Monitor, Wifi, Globe, Cpu } from 'lucide-react';

export default function LiveCamera({ apiUrl }) {
    const [mode, setMode] = useState('host'); // 'host' (laptop) or 'client' (phone)
    const [ws, setWs] = useState(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [isConnected, setIsConnected] = useState(false);
    const [viewerImage, setViewerImage] = useState(null);
    const [results, setResults] = useState(null);
    const [serverInfo, setServerInfo] = useState(null);
    const [loading, setLoading] = useState(true);
    const [useHttpFallback, setUseHttpFallback] = useState(false);
    const [modelConfig, setModelConfig] = useState(null);
    const [availableModels, setAvailableModels] = useState({});

    // Fetch model config on mount
    useEffect(() => {
        const fetchModelConfig = async () => {
            try {
                const response = await fetch(`${window.location.origin}/api/config`);
                if (response.ok) {
                    const data = await response.json();
                    console.log('[LiveCamera] Model config received:', data);
                    setModelConfig(data.config);
                    setAvailableModels(data.available_models || {});
                }
            } catch (error) {
                console.error('[LiveCamera] Failed to fetch model config:', error);
            }
        };
        
        fetchModelConfig();
        // Poll for config changes every 5 seconds
        const interval = setInterval(fetchModelConfig, 5000);
        return () => clearInterval(interval);
    }, []);

    // Fetch server info on mount with retry
    useEffect(() => {
        let retryCount = 0;
        const maxRetries = 3;
        let timeoutId;
        
        const fetchServerInfo = async () => {
            try {
                // Always use same origin - Vite proxy handles routing
                const fetchUrl = window.location.origin;
                
                console.log(`[LiveCamera] Attempt ${retryCount + 1}/${maxRetries} - Fetching from: ${fetchUrl}/api/info`);
                
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
                
                const response = await fetch(`${fetchUrl}/api/info`, {
                    headers: {
                        'Accept': 'application/json',
                    },
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                console.log('[LiveCamera] Server info received:', JSON.stringify(data, null, 2));
                
                // Always set serverInfo regardless of ngrok status
                if (data.frontend_url) {
                    console.log('[LiveCamera] Setting server info with frontend_url:', data.frontend_url);
                    setServerInfo(data);
                    setLoading(false);
                } else {
                    // Fallback - use current origin
                    console.log('[LiveCamera] No frontend_url, using fallback');
                    const fallbackData = {
                        ...data,
                        frontend_url: window.location.origin,
                        has_remote_access: window.location.hostname.includes('ngrok')
                    };
                    setServerInfo(fallbackData);
                    setLoading(false);
                }
            } catch (error) {
                console.error(`[LiveCamera] Fetch failed (attempt ${retryCount + 1}/${maxRetries}):`, error);
                
                retryCount++;
                if (retryCount < maxRetries) {
                    console.log(`[LiveCamera] Retrying in 1 second...`);
                    timeoutId = setTimeout(fetchServerInfo, 1000);
                } else {
                    // Max retries reached - still set serverInfo to allow connection
                    console.warn('[LiveCamera] Max retries reached, using fallback config');
                    setServerInfo({
                        frontend_url: window.location.origin,
                        has_remote_access: window.location.hostname.includes('ngrok'),
                        local_ip: null,
                        error: `Connection issues (${error.message})`
                    });
                    setLoading(false);
                }
            }
        };
        
        fetchServerInfo();
        
        return () => {
            if (timeoutId) clearTimeout(timeoutId);
        };
    }, []);

    // Check if we are on mobile
    useEffect(() => {
        const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
        console.log('[LiveCamera] Device detection - isMobile:', isMobile, 'userAgent:', navigator.userAgent);
        if (isMobile) {
            console.log('[LiveCamera] Setting mode to client (phone)');
            setMode('client');
        } else {
            console.log('[LiveCamera] Setting mode to host (laptop)');
        }
    }, []);

    // Skip WebSocket on phone - go straight to HTTP for reliability through ngrok
    useEffect(() => {
        console.log('[LiveCamera] Connection effect triggered - serverInfo:', serverInfo ? 'present' : 'null', 'mode:', mode);
        
        if (!serverInfo) {
            console.log('[LiveCamera] Skipping - no serverInfo yet');
            return;
        }

        if (mode === 'client') {
            // Phone: Always use HTTP mode (more reliable through ngrok)
            console.log('[Phone] Using HTTP mode (bypassing WebSocket for ngrok compatibility)');
            setUseHttpFallback(true);
            setIsConnected(true);
            return;
        }

        // Laptop (viewer): Try WebSocket first
        const role = 'viewer';
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/api/ws/stream?role=${role}`;

        console.log(`[Viewer] Creating WebSocket connection:`, wsUrl);
        
        let socket;
        let connectionTimeout;
        
        try {
            socket = new WebSocket(wsUrl);
            
            // Set connection timeout
            connectionTimeout = setTimeout(() => {
                if (socket.readyState !== WebSocket.OPEN) {
                    console.warn('[Viewer] WebSocket timeout, will use HTTP polling');
                    socket.close();
                }
            }, 5000);
            
        } catch (error) {
            console.error('[Viewer] Failed to create WebSocket:', error);
            return;
        }

        socket.onopen = () => {
            console.log(`[Viewer] WebSocket CONNECTED`);
            clearTimeout(connectionTimeout);
            setIsConnected(true);
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Handle model change notifications
            if (data.type === 'model_changed') {
                console.log('[Viewer] Model changed:', data.config);
                setModelConfig(data.config);
                return;
            }
            
            // Handle normal frame data
            setResults(data);
            setViewerImage(data.image);
        };

        socket.onerror = (error) => {
            console.error('[Viewer] WebSocket ERROR:', error);
        };

        socket.onclose = (event) => {
            console.log('[Viewer] WebSocket CLOSED');
            clearTimeout(connectionTimeout);
            setIsConnected(false);
        };

        setWs(socket);

        return () => {
            clearTimeout(connectionTimeout);
            if (socket) {
                socket.close();
            }
        };
    }, [mode, serverInfo]);

    // Camera streaming logic (Client only)
    useEffect(() => {
        let interval;
        let streamStarted = false;
        let isProcessing = false; // Throttle flag to prevent concurrent requests

        const startStreaming = async () => {
            console.log('[LiveCamera] startStreaming called - mode:', mode, 'isConnected:', isConnected, 'useHttpFallback:', useHttpFallback);
            
            // Test backend connectivity before starting stream
            if (mode === 'client' && !streamStarted) {
                console.log('[Phone] Testing backend connectivity...');
                try {
                    const healthCheck = await fetch(`${window.location.origin}/api/health`, {
                        method: 'GET',
                        headers: { 'Accept': 'application/json' }
                    });
                    const healthData = await healthCheck.json();
                    console.log('[Phone] ✓ Backend is reachable:', healthData);
                } catch (e) {
                    console.error('[Phone] ✗ Cannot reach backend:', e);
                    console.error('[Phone] Make sure you clicked "Visit Site" on the ngrok warning page!');
                    // Don't return - still try to connect
                }
            }
            
            if (mode === 'client' && isConnected && videoRef.current) {
                try {
                    console.log('[Phone] Requesting camera access...');
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { 
                            facingMode: 'environment',
                            width: { ideal: 640 }, // Reduce resolution to ensure reliability
                            height: { ideal: 480 }
                        }
                    });
                    
                    videoRef.current.srcObject = stream;
                    console.log('[Phone] Camera stream obtained');
                    
                    // Wait for video to be ready
                    videoRef.current.onloadedmetadata = () => {
                        console.log('[Phone] Video metadata loaded, starting frame capture...');
                        videoRef.current.play();
                        
                        // Start sending frames after video is ready
                        setTimeout(() => {
                            interval = setInterval(async () => {
                                // Skip if still processing previous frame
                                if (isProcessing) {
                                    console.log('[Phone] Skipping frame - still processing previous one');
                                    return;
                                }
                                
                                if (videoRef.current && canvasRef.current) {
                                    const video = videoRef.current;
                                    
                                    // Check if video has valid dimensions
                                    if (video.videoWidth === 0 || video.videoHeight === 0) {
                                        console.warn('[Phone] Video not ready yet, skipping frame');
                                        return;
                                    }
                                    
                                    isProcessing = true; // Lock to prevent concurrent sends
                                    
                                    try {
                                        const context = canvasRef.current.getContext('2d');
                                        canvasRef.current.width = video.videoWidth;
                                        canvasRef.current.height = video.videoHeight;
                                        context.drawImage(video, 0, 0);

                                        const base64 = canvasRef.current.toDataURL('image/jpeg', 0.5); // High compression to keep payload small
                                        
                                        const isFirstFrame = !streamStarted;
                                        if (isFirstFrame) {
                                            console.log('[Phone] First frame sent, dimensions:', video.videoWidth, 'x', video.videoHeight);
                                            console.log('[Phone] Using', useHttpFallback ? 'HTTP fallback' : 'WebSocket');
                                            streamStarted = true;
                                        }
                                        
                                    if (useHttpFallback) {
                                        // HTTP fallback: POST to backend with TIMEOUT
                                        const sendTime = Date.now();
                                        console.log('[Phone] Sending frame via HTTP...');
                                        try {
                                            // Create abort controller for timeout
                                            const controller = new AbortController();
                                            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
                                            
                                            const response = await fetch(`${window.location.origin}/api/frame`, {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({ image: base64 }),
                                                signal: controller.signal
                                            });
                                            
                                            clearTimeout(timeoutId);
                                            const result = await response.json();
                                            const elapsed = Date.now() - sendTime;
                                            console.log(`[Phone] HTTP response in ${elapsed}ms:`, result);
                                            
                                            if (result.status === 'busy') {
                                                console.warn('[Phone] Backend busy, will retry next interval');
                                            } else if (result.error) {
                                                console.error('[Phone] Backend error:', result.error);
                                            }
                                        } catch (e) {
                                            const elapsed = Date.now() - sendTime;
                                            if (e.name === 'AbortError') {
                                                console.error(`[Phone] HTTP timeout after ${elapsed}ms - server too slow`);
                                            } else {
                                                console.error(`[Phone] HTTP POST failed after ${elapsed}ms:`, e);
                                            }
                                        }
                                        } else if (ws && ws.readyState === WebSocket.OPEN) {
                                            // WebSocket
                                            ws.send(base64);
                                        }
                                    } finally {
                                        isProcessing = false; // Release lock
                                    }
                                }
                            }, 500); // Reduced to 2 FPS for better inference performance
                        }, 500); // Wait 500ms for video to stabilize
                    };

                } catch (err) {
                    console.error("[Phone] Error accessing camera:", err);
                    alert('Failed to access camera. Please allow camera permissions.');
                }
            }
        };

        startStreaming();

        return () => {
            if (interval) clearInterval(interval);
            if (videoRef.current && videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            }
        };
    }, [mode, isConnected, ws, useHttpFallback]);
    
    // HTTP polling for viewer when phone uses fallback OR WebSocket fails
    useEffect(() => {
        if (mode !== 'host') return;
        
        let pollInterval;
        
        // Start polling immediately if not connected, or after a delay if connected
        const startPolling = () => {
            if (pollInterval) return;
            
            console.log('[Viewer] Starting HTTP polling...');
            pollInterval = setInterval(async () => {
                // If we have a healthy WebSocket connection receiving data, skip polling
                // (We check if we received a frame recently via WS?)
                // For now, we'll just poll as a backup if we don't have an image or if isConnected is false
                
                    try {
                        const response = await fetch(`${window.location.origin}/api/latest`);
                        if (!response.ok) return;
                        
                        const data = await response.json();
                        
                        if (data.image) {
                            setResults(data);
                            setViewerImage(data.image);
                        }
                    } catch (e) {
                        console.error('[Viewer] HTTP poll error:', e);
                    }
            }, 1000); // Poll every 1s (slower than WS)
        };

        // If not connected via WebSocket, poll immediately
        if (!isConnected) {
            startPolling();
        } else {
            // If connected, wait a bit to see if we get data via WS, otherwise poll
            const checkTimeout = setTimeout(() => {
                if (!viewerImage) {
                    console.log('[Viewer] Connected but no data, enabling polling fallback');
                    startPolling();
                }
            }, 3000);
            
            return () => clearTimeout(checkTimeout);
        }
        
        return () => {
            if (pollInterval) clearInterval(pollInterval);
        };
    }, [mode, isConnected, viewerImage]);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="text-center max-w-md">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                    <p className="text-gray-400 mb-2">Initializing connection...</p>
                    <p className="text-xs text-gray-500">Fetching server configuration</p>
                    <div className="mt-4 text-xs text-left bg-gray-800 p-3 rounded">
                        <p className="text-gray-400">Debug Info:</p>
                        <p className="text-gray-500">Origin: {window.location.origin}</p>
                        <p className="text-gray-500">Host: {window.location.host}</p>
                        <p className="text-gray-500">Protocol: {window.location.protocol}</p>
                    </div>
                </div>
            </div>
        );
    }

    const handleModelSwitch = async (modelType) => {
        try {
            console.log('[Viewer] Switching to model:', modelType);
            const response = await fetch(`${window.location.origin}/api/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: modelType })
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('[Viewer] Model switched:', data);
                setModelConfig(data.config);
            } else {
                console.error('[Viewer] Failed to switch model:', response.status);
            }
        } catch (error) {
            console.error('[Viewer] Error switching model:', error);
        }
    };

    if (mode === 'host') {
        // Determine the URL to show in QR code
        // ONLY use ngrok URL - no localhost fallback
        const frontendUrl = serverInfo?.frontend_url || serverInfo?.qr_code_url;
        
        // If no ngrok URL, show error
        if (!frontendUrl) {
            return (
                <div className="flex flex-col items-center justify-center p-8 bg-red-900/20 border border-red-700 rounded-xl">
                    <div className="text-red-400 text-center">
                        <h2 className="text-2xl font-bold mb-4">Ngrok Not Configured</h2>
                        <p className="mb-4">Remote access requires ngrok configuration.</p>
                        <div className="bg-gray-900 p-4 rounded-lg text-left max-w-2xl">
                            <p className="text-sm text-gray-300 mb-2"><strong>Step 1:</strong> Get free ngrok token</p>
                            <p className="text-xs text-blue-400 mb-3">https://dashboard.ngrok.com/signup</p>
                            
                            <p className="text-sm text-gray-300 mb-2"><strong>Step 2:</strong> Add to .env file</p>
                            <code className="text-xs text-green-400 block mb-3">NGROK_AUTHTOKEN=your_token_here</code>
                            
                            <p className="text-sm text-gray-300 mb-2"><strong>Step 3:</strong> Restart services</p>
                            <code className="text-xs text-yellow-400 block">docker-compose down && docker-compose up</code>
                        </div>
                        {serverInfo?.error && (
                            <p className="text-xs text-red-300 mt-4">Error: {serverInfo.error}</p>
                        )}
                    </div>
                </div>
            );
        }

        return (
            <div className="space-y-6">
                {/* Model Selector */}
                {modelConfig && Object.keys(availableModels).length > 0 && (
                    <div className="bg-gray-800 rounded-xl p-4">
                        <div className="flex items-center gap-3 mb-3">
                            <Cpu size={20} className="text-blue-400" />
                            <h3 className="text-lg font-bold">Detection Model</h3>
                        </div>
                        <div className="grid grid-cols-3 gap-3">
                            {Object.entries(availableModels).map(([key, model]) => (
                                <button
                                    key={key}
                                    onClick={() => handleModelSwitch(key)}
                                    className={`p-3 rounded-lg border-2 transition-all ${
                                        modelConfig.type === key
                                            ? 'border-blue-500 bg-blue-900/30 text-blue-300'
                                            : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                                    }`}
                                >
                                    <div className="font-bold text-sm">{model.name}</div>
                                    <div className="text-xs mt-1 opacity-70">{model.description}</div>
                                    {modelConfig.type === key && (
                                        <div className="text-xs mt-2 text-green-400 font-medium">✓ Active</div>
                                    )}
                                </button>
                            ))}
                        </div>
                        <div className="mt-3 text-xs text-gray-400">
                            <strong>Current:</strong> {availableModels[modelConfig.type]?.name || modelConfig.type}
                            {' • '}
                            <strong>Confidence:</strong> {(modelConfig.conf_threshold * 100).toFixed(0)}%
                        </div>
                    </div>
                )}
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="flex flex-col items-center gap-6 text-center p-6 bg-gray-800 rounded-xl">
                    {/* Connection Status Badge */}
                    <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-green-900/30 text-green-400">
                        <Globe size={16} />
                        <span className="text-sm font-medium">Remote Access Enabled</span>
                    </div>

                    <div className="bg-white p-4 rounded-xl">
                        <QRCodeSVG value={frontendUrl} size={200} />
                    </div>
                    
                    <div>
                        <h2 className="text-xl font-bold mb-2">
                            Scan from Anywhere
                        </h2>
                        <p className="text-gray-400 text-sm mb-2">
                            Scan this QR code with your phone from <strong>anywhere in the world</strong>.
                            <br />
                            Works on any network - Wi-Fi or cellular data!
                        </p>

                        {/* URL Display */}
                        <div className="mt-4 bg-gray-900 p-3 rounded-lg">
                            <p className="text-xs text-gray-400 mb-1">Public URL:</p>
                            <code className="text-xs text-green-400 break-all">{frontendUrl}</code>
                        </div>

                        <div className="mt-4 bg-yellow-900/20 border border-yellow-600 p-3 rounded-lg text-left">
                            <p className="text-xs text-yellow-300 font-bold mb-2">⚠️ IMPORTANT:</p>
                            <p className="text-xs text-gray-300 mb-1">
                                1. On phone: Click <strong className="text-yellow-300">"Visit Site"</strong> when ngrok warning appears
                            </p>
                            <p className="text-xs text-gray-300">
                                2. Allow camera when prompted
                            </p>
                        </div>
                        
                        <div className="mt-2 bg-green-900/20 border border-green-700 p-3 rounded-lg">
                            <p className="text-xs text-green-300">
                                <strong>Ready!</strong> Accessible from anywhere in the world.
                            </p>
                        </div>
                    </div>
                    
                    <div className={`flex items-center gap-2 ${isConnected || viewerImage ? 'text-green-400' : 'text-yellow-400'}`}>
                        <Monitor size={20} />
                        <span>
                            {isConnected ? '✓ Viewer Connected (WS)' : 
                             viewerImage ? '✓ Viewer Connected (HTTP)' : 
                             'Waiting for connection...'}
                        </span>
                    </div>
                </div>

                <div className="bg-black rounded-xl overflow-hidden relative min-h-[400px] flex items-center justify-center border border-gray-700">
                    {viewerImage ? (
                        <div className="relative w-full h-full">
                            <img src={viewerImage} alt="Live Stream" className="w-full h-full object-contain" />
                            {/* Canvas overlay for better alignment */}
                            <CanvasOverlay image={viewerImage} boxes={results?.boxes} labels={results?.labels} scores={results?.scores} />
                            
                            {/* Detection Stats Overlay */}
                            {results && (
                                <div className="absolute top-4 right-4 bg-black/80 text-white px-4 py-2 rounded-lg text-sm backdrop-blur-sm">
                                    <div className="font-bold text-green-400">{results.boxes?.length || 0} Detections</div>
                                    {modelConfig && (
                                        <div className="text-xs text-gray-300 mt-1">
                                            {availableModels[modelConfig.type]?.name || modelConfig.type}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-gray-500 flex flex-col items-center">
                            <Smartphone size={48} className="mb-4 opacity-50" />
                            <p>Waiting for camera stream...</p>
                            {modelConfig && (
                                <p className="text-xs mt-2">
                                    Ready with {availableModels[modelConfig.type]?.name || modelConfig.type}
                                </p>
                            )}
                        </div>
                    )}
                </div>
                </div>
            </div>
        );
    }

    return (
        <div className="relative h-[calc(100vh-200px)] bg-black rounded-lg overflow-hidden">
            {!isConnected && mode === 'client' && (
                <div className="absolute inset-0 bg-black/90 z-50 flex items-center justify-center p-8">
                    <div className="bg-yellow-900/30 border-2 border-yellow-500 rounded-xl p-6 max-w-md text-center">
                        <div className="text-6xl mb-4"></div>
                        <h2 className="text-2xl font-bold text-yellow-300 mb-4">Connection Required</h2>
                        <div className="text-left text-white space-y-3 text-sm mb-4">
                            <p><strong>Step 1:</strong> If you see an ngrok warning page, click <strong>"Visit Site"</strong></p>
                            <p><strong>Step 2:</strong> Allow camera permissions when prompted</p>
                            <p><strong>Step 3:</strong> Wait for "Streaming to Laptop..." message</p>
                        </div>
                        <div className="text-xs text-gray-300 bg-black/50 p-3 rounded mt-4">
                            <p>Status: {useHttpFallback ? 'HTTP Mode Active' : 'Connecting...'}</p>
                            <p className="mt-1">If stuck, refresh this page</p>
                        </div>
                        <a 
                            href="/test.html" 
                            className="mt-4 inline-block bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium"
                        >
                            Run Connection Test
                        </a>
                    </div>
                </div>
            )}
            
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute inset-0 w-full h-full object-cover"
            />
            <canvas
                ref={canvasRef}
                className="hidden"
            />

            <div className="absolute bottom-10 left-0 right-0 flex flex-col items-center gap-4">
                <div className="bg-black/70 px-6 py-3 rounded-full backdrop-blur-sm border-2 border-white/20">
                    <p className="text-white font-medium">
                        {isConnected ? '✓ Streaming to Laptop...' : 'Connecting...'}
                    </p>
                    {useHttpFallback && isConnected && (
                        <p className="text-xs text-yellow-300 mt-1">HTTP Mode</p>
                    )}
                    {modelConfig && (
                        <p className="text-xs text-blue-300 mt-1">
                            <Cpu size={12} className="inline mr-1" />
                            Model: {availableModels[modelConfig.type]?.name || modelConfig.type}
                        </p>
                    )}
                </div>
                <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-yellow-500 animate-pulse'}`} />
                    <span className="text-xs text-white bg-black/50 px-2 py-1 rounded">
                        {isConnected ? (useHttpFallback ? 'HTTP' : 'WebSocket') : 'Waiting...'}
                    </span>
                </div>
            </div>
        </div>
    );
}

function CanvasOverlay({ image, boxes, labels, scores }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        if (canvasRef.current && image) {
            const ctx = canvasRef.current.getContext('2d');
            const img = new Image();
            img.src = image;
            img.onload = () => {
                canvasRef.current.width = img.width;
                canvasRef.current.height = img.height;
                ctx.drawImage(img, 0, 0);

                if (boxes) {
                    boxes.forEach((box, i) => {
                        const [x1, y1, x2, y2] = box;
                        ctx.strokeStyle = '#00ff00';
                        ctx.lineWidth = 4;
                        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                        ctx.fillStyle = '#00ff00';
                        ctx.font = '24px Arial';
                        ctx.fillText(`${labels[i]}: ${scores[i].toFixed(2)}`, x1, y1 - 10);
                    });
                }
            };
        }
    }, [image, boxes, labels, scores]);

    return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-contain" />;
}
