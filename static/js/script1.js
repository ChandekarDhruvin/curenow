import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.module.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('threejs-container').appendChild(renderer.domElement);

// Lighting
const ambientLight = new THREE.AmbientLight(0x1a1a1a, 2);
scene.add(ambientLight);
const pointLight = new THREE.PointLight(0x60a5fa, 5, 100);
pointLight.position.set(10, 15, 10);
scene.add(pointLight);

// Enhanced DNA Helix with Connecting Strands
const helixGroup = new THREE.Group();
const helixRadius = 2.5;
const helixHeight = 25;
const helixMaterial = new THREE.MeshPhongMaterial({ color: 0x39ff14, shininess: 150, emissive: 0x39ff14, emissiveIntensity: 0.3 });
const strandMaterial = new THREE.LineBasicMaterial({ color: 0x39ff14, transparent: true, opacity: 0.5 });

const helixPoints = [];
for (let i = 0; i < helixHeight * 2; i += 0.2) {
    const x1 = helixRadius * Math.cos(i);
    const z1 = helixRadius * Math.sin(i);
    const y1 = i - helixHeight / 2;
    const sphereGeo = new THREE.SphereGeometry(0.25, 16, 16);
    const sphere = new THREE.Mesh(sphereGeo, helixMaterial);
    sphere.position.set(x1, y1, z1);
    helixGroup.add(sphere);
    helixPoints.push(new THREE.Vector3(x1, y1, z1));

    const x2 = helixRadius * Math.cos(i + Math.PI);
    const z2 = helixRadius * Math.sin(i + Math.PI);
    const sphere2 = new THREE.Mesh(sphereGeo, helixMaterial);
    sphere2.position.set(x2, y1, z2);
    helixGroup.add(sphere2);
    helixPoints.push(new THREE.Vector3(x2, y1, z2));

    // Add connecting strands
    const strandGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x1, y1, z1),
        new THREE.Vector3(x2, y1, z2)
    ]);
    const strand = new THREE.Line(strandGeo, strandMaterial);
    helixGroup.add(strand);
}
scene.add(helixGroup);

// Brain-like Structure (Wireframe Sphere with Neural Particles)
const brainGroup = new THREE.Group();
const brainGeo = new THREE.SphereGeometry(3, 32, 32);
const brainMaterial = new THREE.MeshPhongMaterial({ color: 0xff5555, emissive: 0xff5555, emissiveIntensity: 0.5, wireframe: true });
const brain = new THREE.Mesh(brainGeo, brainMaterial);
brain.position.set(10, 0, -5); // Offset to the right of the helix
brainGroup.add(brain);

// Neural Particles inside Brain
const neuralGeo = new THREE.BufferGeometry();
const neuralCount = 200;
const neuralPositions = new Float32Array(neuralCount * 3);
const neuralVelocities = new Float32Array(neuralCount * 3);
for (let i = 0; i < neuralCount * 3; i += 3) {
    const theta = Math.random() * 2 * Math.PI;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = Math.random() * 2.5; // Within brain radius
    neuralPositions[i] = r * Math.sin(phi) * Math.cos(theta) + 10; // Offset to brain position
    neuralPositions[i + 1] = r * Math.sin(phi) * Math.sin(theta);
    neuralPositions[i + 2] = r * Math.cos(phi) - 5;
    neuralVelocities[i] = (Math.random() - 0.5) * 0.02;
    neuralVelocities[i + 1] = (Math.random() - 0.5) * 0.02;
    neuralVelocities[i + 2] = (Math.random() - 0.5) * 0.02;
}
neuralGeo.setAttribute('position', new THREE.BufferAttribute(neuralPositions, 3));
const neuralMaterial = new THREE.PointsMaterial({ color: 0xffff00, size: 0.1, transparent: true, opacity: 0.7 });
const neuralParticles = new THREE.Points(neuralGeo, neuralMaterial);
brainGroup.add(neuralParticles);
scene.add(brainGroup);

// Medical Symbols - Floating Cubes
const cubesGroup = new THREE.Group();
const cubeMaterial = new THREE.MeshStandardMaterial({ color: 0xff3333, emissive: 0xff3333, emissiveIntensity: 0.5 });
for (let i = 0; i < 15; i++) {
    const cubeGeo = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const cube = new THREE.Mesh(cubeGeo, cubeMaterial);
    cube.position.set((Math.random() - 0.5) * 25, (Math.random() - 0.5) * 25, (Math.random() - 0.5) * 25);
    cube.rotation.set(Math.random(), Math.random(), Math.random());
    cubesGroup.add(cube);
}
scene.add(cubesGroup);

// Floating Particles
const particleGeo = new THREE.BufferGeometry();
const particleCount = 1000;
const positions = new Float32Array(particleCount * 3);
const velocities = new Float32Array(particleCount * 3);
for (let i = 0; i < particleCount * 3; i += 3) {
    positions[i] = (Math.random() - 0.5) * 100;
    positions[i + 1] = (Math.random() - 0.5) * 100;
    positions[i + 2] = (Math.random() - 0.5) * 100;
    velocities[i] = (Math.random() - 0.5) * 0.01;
    velocities[i + 1] = (Math.random() - 0.5) * 0.01;
    velocities[i + 2] = (Math.random() - 0.5) * 0.01;
}
particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const particleMaterial = new THREE.PointsMaterial({ color: 0x00b7eb, size: 0.15, transparent: true, opacity: 0.8 });
const particles = new THREE.Points(particleGeo, particleMaterial);
scene.add(particles);

// AI Core - Pulsating Sphere
const aiCoreGeo = new THREE.SphereGeometry(1.5, 32, 32);
const aiCoreMaterial = new THREE.MeshStandardMaterial({ color: 0x60a5fa, emissive: 0x60a5fa, emissiveIntensity: 1.5 });
const aiCore = new THREE.Mesh(aiCoreGeo, aiCoreMaterial);
aiCore.position.set(0, -2, 0);
scene.add(aiCore);

// Background Grid
const gridHelper = new THREE.GridHelper(100, 50, 0x333333, 0x333333);
gridHelper.position.set(0, -20, 0);
scene.add(gridHelper);

camera.position.set(0, 5, 25);
camera.lookAt(0, 0, 0);

// Animation
function animate() {
    requestAnimationFrame(animate);

    helixGroup.rotation.y += 0.01;
    brainGroup.rotation.y += 0.005;
    cubesGroup.rotation.y += 0.005;
    particles.rotation.y += 0.002;

    // Animate background particles
    const pos = particles.geometry.attributes.position.array;
    for (let i = 0; i < particleCount * 3; i += 3) {
        pos[i] += velocities[i];
        pos[i + 1] += velocities[i + 1];
        pos[i + 2] += velocities[i + 2];
        if (Math.abs(pos[i]) > 50) velocities[i] *= -1;
        if (Math.abs(pos[i + 1]) > 50) velocities[i + 1] *= -1;
        if (Math.abs(pos[i + 2]) > 50) velocities[i + 2] *= -1;
    }
    particles.geometry.attributes.position.needsUpdate = true;

    // Animate neural particles
    const neuralPos = neuralParticles.geometry.attributes.position.array;
    for (let i = 0; i < neuralCount * 3; i += 3) {
        neuralPos[i] += neuralVelocities[i];
        neuralPos[i + 1] += neuralVelocities[i + 1];
        neuralPos[i + 2] += neuralVelocities[i + 2];
        const dist = Math.sqrt(
            (neuralPos[i] - 10) ** 2 +
            neuralPos[i + 1] ** 2 +
            (neuralPos[i + 2] + 5) ** 2
        );
        if (dist > 2.5) {
            neuralVelocities[i] *= -1;
            neuralVelocities[i + 1] *= -1;
            neuralVelocities[i + 2] *= -1;
        }
    }
    neuralParticles.geometry.attributes.position.needsUpdate = true;

    // Pulsating AI Core
    aiCore.scale.setScalar(1.2 + 0.1 * Math.sin(Date.now() * 0.005));

    // Rotate cubes
    cubesGroup.children.forEach(cube => {
        cube.rotation.x += 0.02;
        cube.rotation.y += 0.02;
    });

    renderer.render(scene, camera);
}
animate();

// Resize handler
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});