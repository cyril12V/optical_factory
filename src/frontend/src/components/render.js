import * as THREE from 'three';
import {MTLLoader} from 'three/examples/jsm/loaders/MTLLoader';
import {OBJLoader} from 'three/examples/jsm/loaders/OBJLoader';
import * as facemesh from '@tensorflow-models/facemesh';
import * as tf from '@tensorflow/tfjs-core';
// Préchargement des backends pour éviter les erreurs de registry
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

var model;
var glassesObj;
var faceObj;
var triangle;
var bg;
var video;
let renderer;
let camera;
var container;
let scene;
let videoSprite;
let windowWidth = 640;
let windowHeight = 480;
var currentGlassesModelName = null;

function setVideoContent(){
    camera = new THREE.PerspectiveCamera(50, video.videoWidth / video.videoHeight, 1, 5000);

    camera.position.z = video.videoHeight;
    camera.position.x = -video.videoWidth / 2;
    camera.position.y = -video.videoHeight / 2;

    bg = new THREE.Texture(video);

    bg.minFilter = THREE.LinearFilter;

    videoSprite = new THREE.Sprite(new THREE.MeshBasicMaterial({
        map: bg,
        depthWrite: false,
        side: THREE.DoubleSide
    }));
    scene = new THREE.Scene();

    scene.add(videoSprite);
    videoSprite.center.set(0.5, 0.5);
    videoSprite.scale.set(-video.videoWidth, video.videoHeight, 1);
    videoSprite.position.copy(camera.position);
    videoSprite.position.z = 0;
}

function setTriangleToScene(){
    const triGeo = new THREE.Geometry();
    triGeo.vertices.push(new THREE.Vector3(1, 0, 0));
    triGeo.vertices.push(new THREE.Vector3(-1, 0, 0));
    triGeo.vertices.push(new THREE.Vector3(0, 0, 1));

    triGeo.faces.push(new THREE.Face3(0, 1, 2));

    triangle = new THREE.Mesh(triGeo, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
    triangle.visible = false;
    scene.add(triangle);
}

function setTheLights(){
    var light = new THREE.PointLight(0xeeeeee);
    light.position.set(10, 50, 20);
    scene.add(light);
     
    var lightAmb = new THREE.AmbientLight(0xff77ff);
    scene.add(lightAmb);
}

function setGlassesToScene(objName){
    if (glassesObj && glassesObj.name) {
        const existingGlasses = scene.getObjectByName(glassesObj.name);
        if (existingGlasses) {
            scene.remove(existingGlasses);
        }
    }
    glassesObj = null;
    currentGlassesModelName = objName;

    // Tableau de tous les chemins possibles à essayer
    const possiblePaths = [
        "/obj/",
        "/src/frontend/obj/",
        "/src/frontend/public/obj/",
        "https://optical-factory-front.vercel.app/obj/",
        "https://optical-factory-front.vercel.app/src/frontend/obj/",
        "/public/obj/"
    ];
    
    // Fonction pour charger avec le chemin de base correct
    const tryLoad = (index) => {
        if (index >= possiblePaths.length) {
            console.error("Tous les chemins ont échoué pour charger", objName);
            return;
        }
        
        const basePath = possiblePaths[index];
        console.log(`Essai de chargement depuis ${basePath}${objName}.mtl`);
        
        const mtlLoader = new MTLLoader();
        mtlLoader.setMaterialOptions({side:THREE.DoubleSide});
        mtlLoader.load(basePath + objName + '.mtl', 
            // Succès
            materials => {
                materials.preload();
                const objLoader = new OBJLoader();
                objLoader.setMaterials(materials);
                objLoader.load(basePath + objName + '.obj', 
                    // Succès
                    obj => {
                        if (currentGlassesModelName === objName) {
                            glassesObj = obj;
                            glassesObj.name = objName;
                            glassesObj.renderOrder = 3;
                            glassesObj.visible = false;
                            scene.add(glassesObj);
                            console.log("SUCCÈS: Chargé depuis", basePath);
                        } else {
                            console.log("Skipping assignment for", objName);
                        }
                    },
                    undefined,
                    // Erreur obj
                    (error) => {
                        console.log(`Échec de chargement ${basePath}${objName}.obj:`, error);
                        tryLoad(index + 1);
                    }
                );
            },
            undefined,
            // Erreur mtl
            (error) => {
                console.log(`Échec de chargement ${basePath}${objName}.mtl:`, error);
                tryLoad(index + 1);
            }
        );
    };
    
    // Démarrer avec le premier chemin
    tryLoad(0);
}

function getFaceMask(){
    // Mêmes chemins possibles que pour les lunettes
    const possiblePaths = [
        "/obj/",
        "/src/frontend/obj/",
        "/src/frontend/public/obj/",
        "https://optical-factory-front.vercel.app/obj/",
        "https://optical-factory-front.vercel.app/src/frontend/obj/",
        "/public/obj/"
    ];
    
    // Fonction pour essayer de charger le masque facial
    const tryLoadFaceMask = (index) => {
        if (index >= possiblePaths.length) {
            console.error("Tous les chemins ont échoué pour charger facemesh.obj");
            return;
        }
        
        const basePath = possiblePaths[index];
        console.log(`Essai de chargement facemesh depuis ${basePath}facemesh.obj`);
        
        new OBJLoader().load(
            basePath + 'facemesh.obj', 
            // Succès
            obj => {
                obj.traverse(child => {
                    if (child instanceof THREE.Mesh) {
                        faceObj = new THREE.Mesh(child.geometry, new THREE.MeshLambertMaterial({side: THREE.FrontSide, color: "blue"}));
                        faceObj.material.colorWrite = false;
                        faceObj.renderOrder = 5;
                        scene.add(faceObj);
                        console.log("SUCCÈS: facemesh chargé depuis", basePath);
                    }
                });
            },
            undefined,
            // Erreur
            (error) => {
                console.log(`Échec de chargement ${basePath}facemesh.obj:`, error);
                tryLoadFaceMask(index + 1);
            }
        );
    };
    
    // Démarrer avec le premier chemin
    tryLoadFaceMask(0);
}

export function IntializeThreejs(objName) {
    video = document.getElementById('tryon-video');

    container = document.getElementById('threejsContainer');
    setVideoContent();

    setTheLights();

    setTriangleToScene();

    getFaceMask();
    
    setGlassesToScene(objName);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(windowWidth, windowWidth * video.videoHeight / video.videoWidth);
    
    container.appendChild(renderer.domElement);
    window.addEventListener('resize', onWindowResize, false);
    animate();
}

function onWindowResize() {
    camera.aspect = video.videoWidth / video.videoHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(windowWidth, windowWidth * video.videoHeight / video.videoWidth);
    renderer.setClearColor( 0xeeeeee, 1 );
}

function animate() {
    bg.needsUpdate = true;
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

export async function IntializeEngine() {
    try {
        // Chargement des backends dans un ordre de préférence
        await Promise.all([
            tf.ready(),
            import('@tensorflow/tfjs-backend-webgl'),
            import('@tensorflow/tfjs-backend-cpu')
        ]);
        
        // Essayer webgl d'abord, sinon utiliser cpu
        try {
            await tf.setBackend('webgl');
            console.log("Backend TensorFlow: webgl");
        } catch (e) {
            console.warn("WebGL non disponible, utilisation de CPU:", e);
            await tf.setBackend('cpu');
            console.log("Backend TensorFlow: cpu");
        }
        
        model = await facemesh.load({ maxFaces: 1 });
        renderPrediction();
    } catch (error) {
        console.error("Erreur d'initialisation TensorFlow:", error);
    }
}

export function changeGlassesModel(newModelName) {
    if (newModelName && newModelName !== currentGlassesModelName) {
        console.log("Requesting change to glasses model:", newModelName);
        setGlassesToScene(newModelName);
    } else {
        console.log("Model", newModelName, "is already loaded or invalid.");
    }
}

async function renderPrediction() {
    
    const predictions = await model.estimateFaces(video);

    if (predictions.length > 0 && glassesObj) {

        faceObj.visible = true;
        glassesObj.visible = true;
        for (let i = 0; i < predictions.length; i++) {
            const points = predictions[i].scaledMesh;

            const v2 = new THREE.Vector3(-points[7][0], -points[7][1], -points[7][2]);
            const v1 = new THREE.Vector3(-points[175][0], -points[175][1], -points[175][2])
            const v3 = new THREE.Vector3(-points[263][0], -points[263][1], -points[263][2])

            triangle.geometry.vertices[0].copy(v1);
            triangle.geometry.vertices[1].copy(v2);
            triangle.geometry.vertices[2].copy(v3);
            triangle.geometry.verticesNeedUpdate = true;
            triangle.geometry.computeFaceNormals();

            const p1 = new THREE.Vector3(-points[10][0], -points[10][1], -points[10][2]);
            const p2 = new THREE.Vector3(-points[175][0], -points[175][1], -points[175][2]);
            const scaleFactor = p1.distanceTo(p2) / 110.5;

            const faceBasePos = new THREE.Vector3(-points[168][0], -points[5][1], -points[10][2]-190);
            const basePosition = new THREE.Vector3(-points[168][0], -points[1][1], -points[10][2]-160);
            const lkt = triangle.geometry.faces[0].normal.clone();
            lkt.transformDirection(triangle.matrixWorld);
            lkt.add(basePosition);

            const lktFace = triangle.geometry.faces[0].normal.clone();
            lktFace.transformDirection(triangle.matrixWorld);
            lktFace.add(faceBasePos);

            faceObj.position.set(faceBasePos.x, faceBasePos.y, faceBasePos.z);
            glassesObj.position.set(basePosition.x, basePosition.y, basePosition.z);
            faceObj.lookAt(lktFace);
            glassesObj.lookAt(lkt);
            
            const diffPosX = basePosition.x - camera.position.x;
            const diffPosY = basePosition.y - camera.position.y;
            const posFactor = windowWidth*3.5 / 800;
            const rotFactor = windowWidth * 1.50;
            const posFactorY = windowHeight*4.5 / 600;
            const rotFactorY = windowHeight*1200 / 600;

            faceObj.position.x += diffPosX / posFactor;
            glassesObj.position.x += diffPosX / posFactor;
            faceObj.rotation.y += -diffPosX/ rotFactor;
            glassesObj.rotation.y += -diffPosX/ rotFactor;

            faceObj.position.y += diffPosY / posFactorY;
            glassesObj.position.y += diffPosY / posFactorY;
            faceObj.rotation.x += diffPosY/ rotFactorY;
            glassesObj.rotation.x += diffPosY/ rotFactorY;

            glassesObj.rotation.x += 0.10;

            if(Math.abs(glassesObj.rotation.y) > 0.15){
                faceObj.position.z -= 40;
                glassesObj.position.z -= 40;
            }

            glassesObj.scale.set(scaleFactor,scaleFactor,scaleFactor*1.35);
            const faceScale = scaleFactor * 1.10;
            faceObj.scale.set(faceScale,faceScale,faceScale);

            faceObj.updateWorldMatrix();
            glassesObj.updateWorldMatrix();

        }
    }
    else if (glassesObj) {
        glassesObj.visible = false;
        faceObj.visible = false;
    }
    requestAnimationFrame(renderPrediction);
}
