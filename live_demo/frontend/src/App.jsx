import React, { useState, useEffect } from 'react';
import VideoInference from './components/VideoInference';
import LiveCamera from './components/LiveCamera';
import { Camera, Video, Settings } from 'lucide-react';

// Use window.location.origin for client-side API calls (browser)
// This works both for local (localhost:3000) and remote (ngrok) access
// Vite proxy will forward /api/* requests to backend
const getApiUrl = () => {
    // Always use same origin - Vite proxy handles routing to backend
    return window.location.origin;
};

function App() {
    const [activeTab, setActiveTab] = useState('video');
    const [config, setConfig] = useState({
        type: 'rcnn',
        checkpoint: 'checkpoints/best_model_rcnn_v2.pth',
        classes: [0, 1, 3]
    });
    const [apiUrl, setApiUrl] = useState(getApiUrl());

    // No need to fetch API URL - we always use same origin with proxy
    useEffect(() => {
        // Just log for debugging
        console.log('App using API URL:', apiUrl);
    }, [apiUrl]);

    const updateConfig = async (newConfig) => {
        const updated = { ...config, ...newConfig };
        setConfig(updated);
        try {
            await fetch(`${apiUrl}/api/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updated)
            });
        } catch (e) {
            console.error("Failed to update config", e);
        }
    };

    return (
        <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-blue-500/30">
            <header className="fixed top-0 left-0 right-0 z-50 bg-black/50 backdrop-blur-xl border-b border-white/5">
                <div className="container mx-auto px-6 h-16 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-gradient-to-tr from-blue-400 to-purple-500"></div>
                        <h1 className="text-lg font-medium tracking-tight text-gray-200">
                            Urban<span className="opacity-50">Sense</span>
                        </h1>
                    </div>

                    <div className="flex gap-1 bg-white/5 p-1 rounded-full border border-white/5">
                        <button
                            onClick={() => setActiveTab('video')}
                            className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-sm transition-all duration-300 ${activeTab === 'video'
                                ? 'bg-white/10 text-white shadow-sm'
                                : 'text-gray-500 hover:text-gray-300'
                                }`}
                        >
                            <Video size={14} />
                            <span>Video</span>
                        </button>
                        <button
                            onClick={() => setActiveTab('live')}
                            className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-sm transition-all duration-300 ${activeTab === 'live'
                                ? 'bg-white/10 text-white shadow-sm'
                                : 'text-gray-500 hover:text-gray-300'
                                }`}
                        >
                            <Camera size={14} />
                            <span>Live</span>
                        </button>
                    </div>

                    <div className="w-24 flex justify-end">
                        {/* Placeholder for future settings or profile */}
                        <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center border border-white/5">
                            <Settings size={14} className="text-gray-500" />
                        </div>
                    </div>
                </div>
            </header>

            <main className="container mx-auto px-6 pt-24 pb-6 h-screen flex flex-col">
                <div className="flex-1 relative">
                    {activeTab === 'video' ? (
                        <div className="bg-white/5 rounded-3xl border border-white/10 p-6 h-full overflow-y-auto custom-scrollbar">
                            <VideoInference apiUrl={apiUrl} />
                        </div>
                    ) : (
                        <LiveCamera apiUrl={apiUrl} />
                    )}
                </div>
            </main>

            <style>{`
                .custom-scrollbar::-webkit-scrollbar {
                    width: 6px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: transparent;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background-color: rgba(255, 255, 255, 0.1);
                    border-radius: 20px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background-color: rgba(255, 255, 255, 0.2);
                }
            `}</style>
        </div>
    );
}

export default App;
