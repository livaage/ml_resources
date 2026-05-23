let scene, camera, renderer, controls;
let surface, point, trail;
let isAnimating = true;
let currentPosition = { x: 2, y: 2, z: 0 };
let learningRate = 0.01;
let trailPoints = [];
let convergenceThreshold = 0.0001;
let stepsSinceLastSignificantChange = 0;
let lastValue = Infinity;

function init() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f5f5);

    // Create camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(8, 8, 8);
    camera.lookAt(0, 0, 0);

    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(document.getElementById('gradient-descent-viz').offsetWidth, 
                   document.getElementById('gradient-descent-viz').offsetHeight);
    document.getElementById('gradient-descent-viz').appendChild(renderer.domElement);

    // Add orbit controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Create surface
    const geometry = new THREE.PlaneGeometry(8, 8, 150, 150);
    
    // Create vertex colors based on height
    const colors = [];
    const positions = geometry.attributes.position.array;
    for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 2];
        const z = calculateZ(x, y);
        positions[i + 1] = z;

        // Create color based on height - using a heat map style gradient
        const normalizedHeight = (z - 2) / 3; // Adjusted normalization range to cover all heights
        const color = new THREE.Color();
        
        if (normalizedHeight < 0.2) {
            color.setStyle('#1a237e'); // Deep indigo
        } else if (normalizedHeight < 0.4) {
            color.setStyle('#0d47a1'); // Deep blue
        } else if (normalizedHeight < 0.6) {
            color.setStyle('#00bcd4'); // Cyan
        } else if (normalizedHeight < 0.8) {
            color.setStyle('#ff9800'); // Orange
        } else {
            color.setStyle('#f44336'); // Red
        }
        
        colors.push(color.r, color.g, color.b);
    }
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        shininess: 100,
        flatShading: false,
        transparent: true,
        opacity: 0.9
    });
    surface = new THREE.Mesh(geometry, material);
    surface.rotation.x = -Math.PI / 2;
    scene.add(surface);

    // Add contour lines
    const contourLevels = [2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6]; // Adjusted contour levels
    const contourResolution = 100;
    
    contourLevels.forEach(level => {
        const points = [];
        for (let i = -4; i <= 4; i += 0.08) {
            for (let j = -4; j <= 4; j += 0.08) {
                const z = calculateZ(i, j);
                if (Math.abs(z - level) < 0.05) {
                    points.push(new THREE.Vector3(i, level, j));
                }
            }
        }
        
        const contourGeometry = new THREE.BufferGeometry().setFromPoints(points);
        const normalizedHeight = (level - 2) / 3; // Adjusted normalization range to match surface
        const color = new THREE.Color();
        
        if (normalizedHeight < 0.2) {
            color.setStyle('#1a237e'); // Deep indigo
        } else if (normalizedHeight < 0.4) {
            color.setStyle('#0d47a1'); // Deep blue
        } else if (normalizedHeight < 0.6) {
            color.setStyle('#00bcd4'); // Cyan
        } else if (normalizedHeight < 0.8) {
            color.setStyle('#ff9800'); // Orange
        } else {
            color.setStyle('#f44336'); // Red
        }
        
        const contourMaterial = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.4
        });
        const contour = new THREE.Points(contourGeometry, contourMaterial);
        scene.add(contour);
    });

    // Add 2D contour map at the bottom
    const contourMapGeometry = new THREE.PlaneGeometry(8, 8, 1, 1);
    const contourMapMaterial = new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide
    });
    const contourMap = new THREE.Mesh(contourMapGeometry, contourMapMaterial);
    contourMap.position.y = 0;
    contourMap.rotation.x = -Math.PI / 2;
    scene.add(contourMap);

    // Adjust camera position for better view
    camera.position.set(8, 8, 8);
    camera.lookAt(0, 0, 0);

    // Add point with trail
    const pointGeometry = new THREE.SphereGeometry(0.1, 32, 32);
    const pointMaterial = new THREE.MeshPhongMaterial({
        color: 0x000000,
        emissive: 0xff0000,
        emissiveIntensity: 0.5,
        shininess: 100
    });
    point = new THREE.Mesh(pointGeometry, pointMaterial);
    scene.add(point);

    // Initialize trail
    const trailGeometry = new THREE.BufferGeometry();
    const trailMaterial = new THREE.LineBasicMaterial({
        vertexColors: true,
        linewidth: 3
    });
    trail = new THREE.Line(trailGeometry, trailMaterial);
    scene.add(trail);

    // Add lights for better surface visualization
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 5);
    scene.add(directionalLight);

    const backLight = new THREE.DirectionalLight(0xffffff, 0.4);
    backLight.position.set(-5, -5, -5);
    scene.add(backLight);

    // Set up controls
    document.getElementById('learningRate').addEventListener('input', function(e) {
        learningRate = parseFloat(e.target.value);
        document.getElementById('learningRateValue').textContent = learningRate.toFixed(3);
    });

    document.getElementById('startX').addEventListener('input', function(e) {
        currentPosition.x = parseFloat(e.target.value);
        document.getElementById('startXValue').textContent = currentPosition.x.toFixed(1);
        resetVisualization();
    });

    updateSurface();
    animate();
}

function updatePointPosition() {
    const x = currentPosition.x;
    const y = currentPosition.y;
    const z = calculateZ(x, y);
    point.position.set(x, z, y);

    // Update gradient arrow
    const gradient = calculateGradient(x, y);
    const gradientMagnitude = Math.sqrt(gradient.x * gradient.x + gradient.y * gradient.y);
    const arrow = scene.children.find(child => child instanceof THREE.Mesh && child.geometry instanceof THREE.CylinderGeometry);
    
    if (arrow) {
        arrow.position.set(x, z, y);
        arrow.scale.set(1, gradientMagnitude, 1);
        arrow.lookAt(x - gradient.x, z, y - gradient.y);
    }
}

function calculateZ(x, y) {
    // Complex loss function with multiple local minima of different depths
    const term1 = 2 * Math.exp(-(x*x + y*y));  // Center well
    const term2 = 1.5 * Math.exp(-((x-2)*(x-2) + (y-2)*(y-2)));  // Shallower well
    const term3 = 3 * Math.exp(-((x+2)*(x+2) + (y-2)*(y-2)));    // Deeper well
    const term4 = 2.5 * Math.exp(-((x-2)*(x-2) + (y+2)*(y+2)));  // Medium well
    const term5 = 1.8 * Math.exp(-((x+2)*(x+2) + (y+2)*(y+2)));  // Another shallow well
    return 5 - (term1 + term2 + term3 + term4 + term5);
}

function calculateGradient(x, y) {
    // Analytical gradient of the complex loss function
    const dx = -2*x*Math.exp(-(x*x + y*y)) +
              -2*(x-2)*Math.exp(-((x-2)*(x-2) + (y-2)*(y-2))) +
              -2*(x+2)*Math.exp(-((x+2)*(x+2) + (y-2)*(y-2))) +
              -2*(x-2)*Math.exp(-((x-2)*(x-2) + (y+2)*(y+2))) +
              -2*(x+2)*Math.exp(-((x+2)*(x+2) + (y+2)*(y+2)));
    
    const dy = -2*y*Math.exp(-(x*x + y*y)) +
              -2*(y-2)*Math.exp(-((x-2)*(x-2) + (y-2)*(y-2))) +
              -2*(y-2)*Math.exp(-((x+2)*(x+2) + (y-2)*(y-2))) +
              -2*(y+2)*Math.exp(-((x-2)*(x-2) + (y+2)*(y+2))) +
              -2*(y+2)*Math.exp(-((x+2)*(x+2) + (y+2)*(y+2)));
    
    return { x: -dx, y: -dy };
}

function updateSurface() {
    const positions = surface.geometry.attributes.position.array;
    const colors = surface.geometry.attributes.color.array;
    
    for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i];
        const y = positions[i + 2];
        const z = calculateZ(x, y);
        positions[i + 1] = z;

        // Update colors based on new height
        const normalizedHeight = (z - 2) / 3; // Adjusted normalization range to cover all heights
        const color = new THREE.Color();
        
        if (normalizedHeight < 0.2) {
            color.setStyle('#1a237e'); // Deep indigo
        } else if (normalizedHeight < 0.4) {
            color.setStyle('#0d47a1'); // Deep blue
        } else if (normalizedHeight < 0.6) {
            color.setStyle('#00bcd4'); // Cyan
        } else if (normalizedHeight < 0.8) {
            color.setStyle('#ff9800'); // Orange
        } else {
            color.setStyle('#f44336'); // Red
        }
        
        colors[i] = color.r;
        colors[i + 1] = color.g;
        colors[i + 2] = color.b;
    }
    
    surface.geometry.attributes.position.needsUpdate = true;
    surface.geometry.attributes.color.needsUpdate = true;
}

function updateTrail() {
    if (trailPoints.length > 100) {
        trailPoints.shift();
    }
    
    const currentZ = calculateZ(currentPosition.x, currentPosition.y);
    trailPoints.push(new THREE.Vector3(
        currentPosition.x,
        currentZ,
        currentPosition.y
    ));

    const positions = new Float32Array(trailPoints.length * 3);
    const colors = new Float32Array(trailPoints.length * 3);
    
    trailPoints.forEach((point, i) => {
        positions[i * 3] = point.x;
        positions[i * 3 + 1] = point.y;
        positions[i * 3 + 2] = point.z;
        
        // Create gradient color for trail
        const t = i / trailPoints.length;
        colors[i * 3] = 1;     // R
        colors[i * 3 + 1] = t; // G
        colors[i * 3 + 2] = t; // B
    });

    trail.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    trail.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    trail.material = new THREE.LineBasicMaterial({ 
        vertexColors: true,
        linewidth: 2
    });
    trail.geometry.attributes.position.needsUpdate = true;
    trail.geometry.attributes.color.needsUpdate = true;
}

function updateStatus() {
    const z = calculateZ(currentPosition.x, currentPosition.y);
    const gradient = calculateGradient(currentPosition.x, currentPosition.y);
    const gradientMagnitude = Math.sqrt(gradient.x * gradient.x + gradient.y * gradient.y);
    
    let status = "Running";
    if (gradientMagnitude < convergenceThreshold) {
        // Check if we're in the deepest well (top-left well at x=-2, y=2)
        const distanceToDeepestWell = Math.sqrt(
            Math.pow(currentPosition.x - (-2), 2) + 
            Math.pow(currentPosition.y - 2, 2)
        );
        
        // If we're close to the deepest well and at a low point, it's the global minimum
        const isGlobalMinimum = distanceToDeepestWell < 0.5 && z < 2.5;
        status = "Converged to " + (isGlobalMinimum ? "global" : "local") + " minimum";
    }

    document.getElementById('statusDisplay').innerHTML = 
        `Position: (${currentPosition.x.toFixed(2)}, ${currentPosition.y.toFixed(2)})<br>` +
        `Value: ${z.toFixed(2)}<br>` +
        `Gradient: (${gradient.x.toFixed(2)}, ${gradient.y.toFixed(2)})<br>` +
        `Status: ${status}`;
}

function animate() {
    requestAnimationFrame(animate);

    if (isAnimating) {
        const gradient = calculateGradient(currentPosition.x, currentPosition.y);
        const gradientMagnitude = Math.sqrt(gradient.x * gradient.x + gradient.y * gradient.y);
        
        if (gradientMagnitude > convergenceThreshold) {
            currentPosition.x -= learningRate * gradient.x;
            currentPosition.y -= learningRate * gradient.y;
            updatePointPosition();
            updateTrail();
        } else {
            isAnimating = false;
        }
    }

    updateStatus();
    controls.update();
    renderer.render(scene, camera);
}

function resetVisualization() {
    currentPosition = { 
        x: parseFloat(document.getElementById('startX').value),
        y: 2,
        z: 0 
    };
    trailPoints = [];
    isAnimating = true;
    updatePointPosition();
    updateTrail();
}

function toggleAnimation() {
    isAnimating = !isAnimating;
}

// Handle window resize
window.addEventListener('resize', () => {
    const container = document.getElementById('gradient-descent-viz');
    camera.aspect = container.offsetWidth / container.offsetHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.offsetWidth, container.offsetHeight);
});

// Initialize visualization when the DOM is loaded
document.addEventListener('DOMContentLoaded', init); 