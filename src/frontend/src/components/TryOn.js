import React, { useEffect, useState } from 'react';
import '../style/TryOn.style.css';
import { IntializeEngine, IntializeThreejs } from './render.js';

export const TryOn = () => {
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        let videoElement = null;
        let cleanupCalled = false;

        async function init() {
            try {
                setIsLoading(true);
                
                // Assurons-nous qu'il n'y a qu'un seul élément vidéo
                const existingVideo = document.getElementById('tryon-video');
                if (existingVideo) {
                    // S'il existe déjà, nettoyons-le d'abord
                    if (existingVideo.srcObject) {
                        const tracks = existingVideo.srcObject.getTracks();
                        tracks.forEach(track => track.stop());
                    }
                    existingVideo.remove();
                }
                
                // Créer un nouvel élément vidéo
                videoElement = document.createElement('video');
                videoElement.id = 'tryon-video';
                videoElement.style.display = 'none';
                videoElement.playsInline = true;
                videoElement.muted = true;
                
                // Ajouter au conteneur
                const container = document.getElementById('threejsContainer');
                if (container) {
                    container.appendChild(videoElement);
                } else {
                    throw new Error("Le conteneur Three.js n'a pas été trouvé");
                }

                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        audio: false,
                        video: {
                            facingMode: 'user',
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                        }
                    });
                    
                    videoElement.srcObject = stream;
                } catch (mediaError) {
                    console.error("Erreur d'accès à la caméra:", mediaError);
                    setError("Impossible d'accéder à votre caméra. Veuillez vérifier vos permissions.");
                    setIsLoading(false);
                    return;
                }

                videoElement.oncanplay = () => {
                    if (!cleanupCalled) {
                        videoElement.play().then(() => {
                            // Une fois la vidéo lancée, initialiser Three.js
                            IntializeThreejs("purple1");
                            IntializeEngine();
                            setIsLoading(false);
                        }).catch(playError => {
                            console.error("Erreur lors de la lecture vidéo:", playError);
                            setError("Impossible de démarrer la vidéo.");
                            setIsLoading(false);
                        });
                    }
                };
            } catch (err) {
                console.error("Erreur d'initialisation:", err);
                setError("Une erreur s'est produite lors de l'initialisation de l'essayage virtuel.");
                setIsLoading(false);
            }
        }

        init();

        // Nettoyage lors du démontage du composant
        return () => {
            cleanupCalled = true;
            
            // Nettoyer la caméra
            if (videoElement && videoElement.srcObject) {
                const tracks = videoElement.srcObject.getTracks();
                tracks.forEach(track => track.stop());
            }
            
            // Supprimer tous les canvas et contenus Three.js
            const container = document.getElementById('threejsContainer');
            if (container) {
                while (container.firstChild) {
                    container.removeChild(container.firstChild);
                }
            }
        };
    }, []);

    return (
        <div className="tryon-container">
            {isLoading && (
                <div className="tryon-loader">
                    <div className="tryon-loader__spinner"></div>
                    <p className="tryon-loader__text">Initialisation de l'essayage virtuel...</p>
                </div>
            )}
            
            {error && (
                <div className="tryon-error">
                    <p className="tryon-error__message">{error}</p>
                    <button className="button button--primary" onClick={() => window.location.reload()}>
                        Réessayer
                    </button>
                </div>
            )}
            
            <div id="threejsContainer" className={isLoading || error ? 'hidden' : ''}>
                {/* L'élément vidéo sera créé dynamiquement via JavaScript */}
                <div className="tryon-overlay">
                    <div className="tryon-instructions">
                        <p>Pour un meilleur résultat, positionnez votre visage dans le cadre et gardez la tête droite</p>
                    </div>
                </div>
            </div>
        </div>
    );
};