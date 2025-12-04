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
        <div className="min-h-screen bg-gray-900 text-white">
            <header className="bg-gray-800 p-4 shadow-lg">
                <div className="container mx-auto flex justify-between items-center">
                    <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                        Urban Issue Detection
                    </h1>
                    <div className="flex gap-4">
                        <select
                            className="bg-gray-700 rounded px-3 py-1"
                            value={config.type}
                            onChange={(e) => updateConfig({ type: e.target.value })}
                        >
                            <option value="rcnn">Faster R-CNN</option>
                            <option value="fcos">FCOS</option>
                            <option value="retinanet">RetinaNet</option>
                        </select>
                    </div>
                </div>
            </header>

            <main className="container mx-auto p-4">
                <div className="flex gap-4 mb-6">
                    <button
                        onClick={() => setActiveTab('video')}
                        className={`flex items-center gap-2 px-6 py-3 rounded-lg transition-all ${activeTab === 'video'
                            ? 'bg-blue-600 shadow-blue-500/30 shadow-lg'
                            : 'bg-gray-800 hover:bg-gray-700'
                            }`}
                    >
                        <Video size={20} />
                        Video Inference
                    </button>
                    <button
                        onClick={() => setActiveTab('live')}
                        className={`flex items-center gap-2 px-6 py-3 rounded-lg transition-all ${activeTab === 'live'
                            ? 'bg-purple-600 shadow-purple-500/30 shadow-lg'
                            : 'bg-gray-800 hover:bg-gray-700'
                            }`}
                    >
                        <Camera size={20} />
                        Live Camera
                    </button>
                </div>

                <div className="bg-gray-800 rounded-xl p-6 shadow-xl border border-gray-700">
                    {activeTab === 'video' ? (
                        <VideoInference apiUrl={apiUrl} />
                    ) : (
                        <LiveCamera apiUrl={apiUrl} />
                    )}
                </div>
            </main>
        </div>
    );
}

export default App;
