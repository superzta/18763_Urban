import React, { useState } from 'react';
import { Upload, Play, Loader2 } from 'lucide-react';

export default function VideoInference({ apiUrl = 'http://localhost:8000' }) {
    const [file, setFile] = useState(null);
    const [processing, setProcessing] = useState(false);
    const [resultUrl, setResultUrl] = useState(null);

    const handleUpload = async () => {
        if (!file) return;

        setProcessing(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${apiUrl}/api/predict_video`, {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                setResultUrl(url);
            } else {
                console.error('Upload failed');
            }
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setProcessing(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="border-2 border-dashed border-gray-600 rounded-xl p-8 text-center hover:border-blue-500 transition-colors">
                <input
                    type="file"
                    accept="video/*"
                    onChange={(e) => setFile(e.target.files[0])}
                    className="hidden"
                    id="video-upload"
                />
                <label htmlFor="video-upload" className="cursor-pointer flex flex-col items-center gap-4">
                    <div className="p-4 bg-gray-700 rounded-full">
                        <Upload size={32} className="text-blue-400" />
                    </div>
                    <div>
                        <p className="text-lg font-medium">Click to upload video</p>
                        <p className="text-gray-400 text-sm">MP4, AVI, MOV supported</p>
                    </div>
                </label>
                {file && (
                    <div className="mt-4 text-blue-400 font-medium">
                        Selected: {file.name}
                    </div>
                )}
            </div>

            <div className="flex justify-center">
                <button
                    onClick={handleUpload}
                    disabled={!file || processing}
                    className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed px-8 py-3 rounded-lg font-bold flex items-center gap-2 transition-all"
                >
                    {processing ? (
                        <>
                            <Loader2 className="animate-spin" />
                            Processing...
                        </>
                    ) : (
                        <>
                            <Play size={20} />
                            Start Inference
                        </>
                    )}
                </button>
            </div>

            {resultUrl && (
                <div className="mt-8">
                    <h3 className="text-xl font-bold mb-4">Result</h3>
                    <video
                        src={resultUrl}
                        controls
                        className="w-full rounded-lg border border-gray-700 shadow-lg"
                    />
                </div>
            )}
        </div>
    );
}
