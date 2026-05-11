const scene = new THREE.Scene(); 
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 
window.innerHeight, 0.1, 1000); 
camera.position.z = 5; 

const renderer = new THREE.WebGLRenderer({
    canvas: document.getElementById('canvas')
});

renderer.setSize(window.innerWidth, window.innerHeight);

const light = new THREE.PointLight(0xffffff, 1); 
light.position.set(10, 10, 10);

scene.add(light); 

const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
scene.add(ambientLight);

const loader = new THREE.GLTFLoader(); 
loader.load('model.glb', function (gltf) { 
    scene.add(gltf.scene);
});

const controls = new THREE.OrbitControls(camera, renderer.domElement);

window.addEventListener('resize', function () { 
    camera.aspect = window.innerWidth / window.innerHeight; 
    camera.updateProjectionMatrix(); 
    renderer.setSize(window.innerWidth, window.innerHeight); 
}, false);

function animate() { 
    requestAnimationFrame(animate); 
    renderer.render(scene, camera); 

} 
animate();