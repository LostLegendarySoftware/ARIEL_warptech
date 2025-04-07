<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dreaming Elon - Mars Propulsion Simulator</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap');
        
        body {
            font-family: 'Orbitron', sans-serif;
            background-color: #0a0a1a;
            color: #fff;
            overflow-x: hidden;
        }
        
        .cyberpunk-border {
            border: 2px solid #5D5CDE;
            box-shadow: 0 0 15px #5D5CDE, inset 0 0 10px #5D5CDE;
        }
        
        .cyberpunk-glow {
            text-shadow: 0 0 10px #5D5CDE;
        }
        
        .neon-button {
            background-color: transparent;
            border: 2px solid #5D5CDE;
            color: #fff;
            text-shadow: 0 0 5px #5D5CDE;
            box-shadow: 0 0 10px #5D5CDE;
            transition: all 0.3s ease;
        }
        
        .neon-button:hover {
            background-color: #5D5CDE;
            box-shadow: 0 0 20px #5D5CDE;
        }

        .planet-circle {
            border-radius: 50%;
            position: absolute;
        }

        /* Dark mode */
        .dark {
            background-color: #0a0a1a;
            color: #fff;
        }

        /* Progress indicator */
        .progress-line {
            height: 4px;
            background: linear-gradient(90deg, #5D5CDE 0%, rgba(93, 92, 222, 0.2) 100%);
            position: relative;
        }

        .ship-indicator {
            position: absolute;
            width: 12px;
            height: 12px;
            background-color: #fff;
            border-radius: 50%;
            transform: translateY(-4px);
            box-shadow: 0 0 10px #fff, 0 0 20px #5D5CDE;
        }

        .thought-bubble {
            position: relative;
            background: rgba(93, 92, 222, 0.2);
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 25px;
            border: 1px solid #5D5CDE;
            box-shadow: 0 0 10px rgba(93, 92, 222, 0.5);
        }
        
        .thought-bubble:after {
            content: '';
            position: absolute;
            bottom: -20px;
            left: 50%;
            width: 20px;
            height: 20px;
            background: rgba(93, 92, 222, 0.2);
            border: 1px solid #5D5CDE;
            border-radius: 50%;
            transform: translateX(-50%);
            box-shadow: 0 0 10px rgba(93, 92, 222, 0.5);
        }
        
        .thought-bubble:before {
            content: '';
            position: absolute;
            bottom: -35px;
            left: 50%;
            width: 10px;
            height: 10px;
            background: rgba(93, 92, 222, 0.2);
            border: 1px solid #5D5CDE;
            border-radius: 50%;
            transform: translateX(-100%);
            box-shadow: 0 0 10px rgba(93, 92, 222, 0.5);
        }

        /* Animation */
        @keyframes pulse {
            0% { opacity: 0.7; }
            50% { opacity: 1; }
            100% { opacity: 0.7; }
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        /* Canvas container */
        #canvas-container {
            width: 100%;
            height: 300px;
            position: relative;
        }

        /* Phase indicators */
        .phase-indicator {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.3);
            margin: 0 5px;
            transition: all 0.3s ease;
        }
        
        .phase-active {
            background-color: #5D5CDE;
            box-shadow: 0 0 10px #5D5CDE;
        }

        /* For dark mode support */
        @media (prefers-color-scheme: dark) {
            body {
                background-color: #0a0a1a;
                color: #fff;
            }
        }
    </style>
</head>
<body>
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <header class="text-center mb-8">
            <h1 class="text-4xl md:text-5xl font-bold cyberpunk-glow mb-2">DREAMING ELON</h1>
            <p class="text-lg text-blue-300">SpaceX Next-Gen Propulsion Concept</p>
            <p class="text-sm text-red-400 mt-1">FOR INTERNAL SPACEX USE ONLY - PROOF OF CONCEPT</p>
        </header>

        <!-- Main simulation area -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Left panel - Elon + Thought Bubble -->
            <div class="cyberpunk-border rounded-lg p-4 relative">
                <div class="flex flex-col items-center">
                    <div class="thought-bubble w-full mb-8">
                        <p id="phase-text" class="text-center text-lg">Initializing systems...</p>
                    </div>
                    
                    <div class="relative w-48 h-48 mx-auto">
                        <div class="w-full h-full rounded-lg overflow-hidden cyberpunk-border bg-gray-900 relative">
                            <!-- SVG representation of Elon Musk -->
                            <svg viewBox="0 0 100 120" class="w-full h-full">
                                <!-- Face shape -->
                                <path d="M25,40 Q50,20 75,40 L75,85 Q50,105 25,85 Z" fill="#f0d0b0" />
                                
                                <!-- Hair -->
                                <path d="M25,40 Q50,20 75,40 L75,30 Q65,15 35,15 Q20,20 25,40 Z" fill="#333333" />
                                <path d="M26,38 C30,32 35,30 40,27 C35,25 28,28 26,38 Z" fill="#333333" />
                                <path d="M74,38 C70,32 65,30 60,27 C65,25 72,28 74,38 Z" fill="#333333" />
                                <path d="M30,25 Q50,15 70,25 L65,18 Q50,10 35,18 Z" fill="#333333" />
                                
                                <!-- Ears -->
                                <path d="M23,58 Q20,65 23,72 Q18,70 17,65 Q16,60 23,58" fill="#f0d0b0" />
                                <path d="M77,58 Q80,65 77,72 Q82,70 83,65 Q84,60 77,58" fill="#f0d0b0" />
                                
                                <!-- Eyes -->
                                <ellipse cx="40" cy="50" rx="6" ry="4" fill="#ffffff" />
                                <ellipse cx="60" cy="50" rx="6" ry="4" fill="#ffffff" />
                                <ellipse cx="40" cy="50" rx="3" ry="3" fill="#483d28" />
                                <ellipse cx="60" cy="50" rx="3" ry="3" fill="#483d28" />
                                <ellipse cx="41" cy="49" rx="1" ry="1" fill="#ffffff" />
                                <ellipse cx="61" cy="49" rx="1" ry="1" fill="#ffffff" />
                                
                                <!-- Eye bags - signature Elon feature -->
                                <path d="M34,54 Q40,58 46,54" fill="none" stroke="#d8b090" stroke-width="1" opacity="0.6" />
                                <path d="M54,54 Q60,58 66,54" fill="none" stroke="#d8b090" stroke-width="1" opacity="0.6" />
                                
                                <!-- Eyebrows -->
                                <path d="M32,43 Q40,40 47,43" fill="none" stroke="#333333" stroke-width="1.5" />
                                <path d="M53,43 Q60,40 68,43" fill="none" stroke="#333333" stroke-width="1.5" />
                                
                                <!-- Nose -->
                                <path d="M50,53 Q53,65 50,70 Q47,65 50,53" fill="#e8c4a0" />
                                
                                <!-- Mouth - slight smirk -->
                                <path d="M40,78 Q50,84 60,78" fill="none" stroke="#8a6a5e" stroke-width="1.5" />
                                <path d="M42,78 Q50,82 58,78" fill="none" stroke="#ca9a8e" stroke-width="0.8" opacity="0.5" />
                                
                                <!-- Facial features -->
                                <path d="M32,58 L35,60" fill="none" stroke="#e8c4a0" stroke-width="0.7" />
                                <path d="M68,58 L65,60" fill="none" stroke="#e8c4a0" stroke-width="0.7" />
                                
                                <!-- Cheeks -->
                                <ellipse cx="35" cy="67" rx="5" ry="4" fill="#f3c0a0" opacity="0.2" />
                                <ellipse cx="65" cy="67" rx="5" ry="4" fill="#f3c0a0" opacity="0.2" />
                                
                                <!-- Chin -->
                                <ellipse cx="50" cy="85" rx="15" ry="8" fill="#e8c4a0" opacity="0.3" />
                                <path d="M45,90 Q50,95 55,90" fill="none" stroke="#d8b090" stroke-width="0.7" opacity="0.6" />
                                
                                <!-- Neck -->
                                <path d="M38,95 L38,105 L62,105 L62,95" fill="#f0d0b0" />
                                
                                <!-- Tesla/SpaceX style t-shirt with logo hint -->
                                <path d="M36,105 L30,110 L70,110 L64,105" fill="#222" />
                                <path d="M50,105 L50,115" fill="none" stroke="#222" stroke-width="2" />
                                <path d="M45,108 L50,106 L55,108" fill="none" stroke="#444" stroke-width="0.8" />
                            </svg>
                        </div>
                        <div class="absolute top-0 right-0 bg-blue-500 text-white px-2 py-1 rounded-full text-xs animate-pulse">
                            hmm...
                        </div>
                    </div>
                    
                    <h3 class="mt-4 text-xl">Current Phase:</h3>
                    <div class="flex justify-center mt-2 mb-4">
                        <div id="phase1" class="phase-indicator phase-active" title="Initialization"></div>
                        <div id="phase2" class="phase-indicator" title="Small Nuclear Reactions"></div>
                        <div id="phase3" class="phase-indicator" title="Ion Thrusters"></div>
                        <div id="phase4" class="phase-indicator" title="Central Reactor"></div>
                        <div id="phase5" class="phase-indicator" title="Light-Speed Push"></div>
                    </div>
                    
                    <div class="w-full mt-4">
                        <div class="flex justify-between text-sm mb-1">
                            <span>Earth</span>
                            <span>Mars</span>
                        </div>
                        <div class="progress-line w-full relative">
                            <div id="ship-progress" class="ship-indicator" style="left: 0%"></div>
                        </div>
                    </div>
                    
                    <div class="mt-6 w-full">
                        <button id="start-btn" class="neon-button py-2 px-4 rounded w-full">START SIMULATION</button>
                        <button id="reset-btn" class="mt-2 bg-transparent border border-red-500 text-red-500 py-2 px-4 rounded w-full hover:bg-red-500 hover:text-white transition-all" style="display:none;">RESET</button>
                    </div>
                </div>
            </div>
            
            <!-- Middle panel - 3D Visualization -->
            <div class="cyberpunk-border rounded-lg p-4 lg:col-span-2">
                <h3 class="text-xl mb-4 text-center">Propulsion System Simulation</h3>
                <div id="canvas-container" class="w-full"></div>
                
                <div class="grid grid-cols-2 gap-4 mt-4">
                    <div>
                        <h4 class="text-lg mb-2">System Status</h4>
                        <div class="grid grid-cols-2 gap-2 text-sm">
                            <div>Speed:</div>
                            <div id="speed-value">0 km/s</div>
                            <div>Temperature:</div>
                            <div id="temp-value">293 K</div>
                            <div>Distance:</div>
                            <div id="distance-value">0 million km</div>
                            <div>Time Elapsed:</div>
                            <div id="time-value">0 days</div>
                        </div>
                    </div>
                    <div>
                        <h4 class="text-lg mb-2">Reactor Output</h4>
                        <div class="grid grid-cols-2 gap-2 text-sm">
                            <div>Small Reactors:</div>
                            <div id="small-reactor">Inactive</div>
                            <div>Central Reactor:</div>
                            <div id="central-reactor">Inactive</div>
                            <div>Ion Thrusters:</div>
                            <div id="ion-thrusters">Inactive</div>
                            <div>Cooling System:</div>
                            <div id="cooling-system">Standby</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Bottom panel - Graphs -->
            <div class="cyberpunk-border rounded-lg p-4 col-span-1 lg:col-span-3">
                <h3 class="text-xl mb-4 text-center">Performance Metrics</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <canvas id="speed-chart" width="400" height="200"></canvas>
                    </div>
                    <div>
                        <canvas id="reactor-chart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="mt-8 text-center text-sm">
            <p class="text-blue-400 font-semibold">LOST LEGENDARY SOFTWARE - A SECOND STAR CO.</p>
            <p class="text-gray-300">NUKLION THRUSTER: NUCLEAR/ION BASED THRUST PROOF OF CONCEPT</p>
            <p class="mt-2 text-xs text-red-400">Note: This is a joke and proof of concept acting as proposition for collaboration with Elon Musk but is not affiliated with SpaceX or Elon Musk!</p>
        </footer>
    </div>

    <script>
        // Check for dark mode preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.documentElement.classList.add('dark');
        }
        
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
            if (event.matches) {
                document.documentElement.classList.add('dark');
            } else {
                document.documentElement.classList.remove('dark');
            }
        });
        
        // Simulation variables
        let isSimulationRunning = false;
        let currentPhase = 1;
        let progress = 0;
        let speed = 0;
        let temperature = 293;
        let distance = 0;
        let timeElapsed = 0;
        let animationId;
        let simulationInterval;
        
        // Phase data
        const phases = [
            { name: "Initialization", description: "SpaceX systems coming online. ARIEL monitoring activated." },
            { name: "Small Nuclear Reactions", description: "Triangular Raptor-X setup initiating sequential nuclear thrust." },
            { name: "Ion Thrusters", description: "Starship's 19 waves of ion thrusters maintaining velocity. Cooling active." },
            { name: "Central Reactor", description: "Mars Orbital Transfer System charging. Heat redistribution in progress." },
            { name: "Light-Speed Push", description: "All systems at maximum output. Approaching Mars in record time!" }
        ];
        
        // DOM Elements
        const startBtn = document.getElementById('start-btn');
        const resetBtn = document.getElementById('reset-btn');
        const shipProgress = document.getElementById('ship-progress');
        const phaseText = document.getElementById('phase-text');
        const phaseIndicators = [
            document.getElementById('phase1'),
            document.getElementById('phase2'),
            document.getElementById('phase3'),
            document.getElementById('phase4'),
            document.getElementById('phase5')
        ];
        
        // Status elements
        const speedValue = document.getElementById('speed-value');
        const tempValue = document.getElementById('temp-value');
        const distanceValue = document.getElementById('distance-value');
        const timeValue = document.getElementById('time-value');
        const smallReactor = document.getElementById('small-reactor');
        const centralReactor = document.getElementById('central-reactor');
        const ionThrusters = document.getElementById('ion-thrusters');
        const coolingSystem = document.getElementById('cooling-system');
        
        // Initialize Three.js scene
        let scene, camera, renderer, propulsionSystem;
        
        function initThreeJs() {
            // Scene setup
            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(75, document.getElementById('canvas-container').offsetWidth / document.getElementById('canvas-container').offsetHeight, 0.1, 1000);
            
            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(document.getElementById('canvas-container').offsetWidth, document.getElementById('canvas-container').offsetHeight);
            renderer.setClearColor(0x0a0a1a, 1);
            
            // Clear any existing canvas
            const container = document.getElementById('canvas-container');
            while (container.firstChild) {
                container.removeChild(container.firstChild);
            }
            container.appendChild(renderer.domElement);
            
            // Add lights
            const ambientLight = new THREE.AmbientLight(0x404040);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // Create propulsion system
            propulsionSystem = new THREE.Group();
            
            // Central large reactor (sphere)
            const centralGeometry = new THREE.SphereGeometry(1, 32, 32);
            const centralMaterial = new THREE.MeshPhongMaterial({
                color: 0x5D5CDE,
                emissive: 0x5D5CDE,
                emissiveIntensity: 0.3,
                transparent: true,
                opacity: 0.8
            });
            const centralReactor = new THREE.Mesh(centralGeometry, centralMaterial);
            propulsionSystem.add(centralReactor);
            
            // Three small reactors in triangle formation
            const smallGeometry = new THREE.SphereGeometry(0.5, 16, 16);
            const smallMaterial = new THREE.MeshPhongMaterial({
                color: 0x00ffff,
                emissive: 0x00ffff,
                emissiveIntensity: 0.3,
                transparent: true,
                opacity: 0.8
            });
            
            // Position the three small reactors in a triangle
            for (let i = 0; i < 3; i++) {
                const angle = (i * Math.PI * 2) / 3;
                const smallReactor = new THREE.Mesh(smallGeometry, smallMaterial);
                smallReactor.position.x = Math.cos(angle) * 2;
                smallReactor.position.z = Math.sin(angle) * 2;
                
                // Add pulsing effect
                smallReactor.userData = { 
                    baseScale: 1,
                    phaseOffset: i * Math.PI * 0.5 
                };
                
                propulsionSystem.add(smallReactor);
            }
            
            // Add ion thrusters (cylinders)
            const thrusterGeometry = new THREE.CylinderGeometry(0.1, 0.2, 1, 16);
            const thrusterMaterial = new THREE.MeshPhongMaterial({
                color: 0xff5500,
                emissive: 0xff5500,
                emissiveIntensity: 0.5,
                transparent: true,
                opacity: 0.7
            });
            
            // Create 19 ion thrusters in a circular pattern
            for (let i = 0; i < 19; i++) {
                const angle = (i * Math.PI * 2) / 19;
                const radius = 3;
                const thruster = new THREE.Mesh(thrusterGeometry, thrusterMaterial);
                thruster.position.x = Math.cos(angle) * radius;
                thruster.position.z = Math.sin(angle) * radius;
                thruster.rotation.x = Math.PI / 2;
                propulsionSystem.add(thruster);
            }
            
            // Add octagonal frame (torus)
            const torusGeometry = new THREE.TorusGeometry(3.5, 0.1, 8, 8);
            const torusMaterial = new THREE.MeshPhongMaterial({
                color: 0xffffff,
                emissive: 0x5D5CDE,
                emissiveIntensity: 0.5
            });
            const torus = new THREE.Mesh(torusGeometry, torusMaterial);
            torus.rotation.x = Math.PI / 2;
            propulsionSystem.add(torus);
            
            scene.add(propulsionSystem);
            
            // Position camera
            camera.position.z = 8;
            camera.position.y = 3;
            camera.lookAt(0, 0, 0);
            
            // Add planets
            const earthGeometry = new THREE.SphereGeometry(0.5, 16, 16);
            const earthMaterial = new THREE.MeshPhongMaterial({
                color: 0x2244aa,
                emissive: 0x001133,
                emissiveIntensity: 0.3
            });
            const earth = new THREE.Mesh(earthGeometry, earthMaterial);
            earth.position.set(-7, 0, 0);
            scene.add(earth);
            
            const marsGeometry = new THREE.SphereGeometry(0.3, 16, 16);
            const marsMaterial = new THREE.MeshPhongMaterial({
                color: 0xdd4422,
                emissive: 0x441100,
                emissiveIntensity: 0.3
            });
            const mars = new THREE.Mesh(marsGeometry, marsMaterial);
            mars.position.set(7, 0, 0);
            scene.add(mars);
            
            animate();
        }
        
        function animate() {
            animationId = requestAnimationFrame(animate);
            
            // Rotate the propulsion system
            if (propulsionSystem) {
                propulsionSystem.rotation.y += 0.005;
                
                // Pulse effect based on current phase
                if (isSimulationRunning) {
                    // Make the elements pulse based on current phase
                    const time = Date.now() * 0.001;
                    
                    // Central reactor pulses in phase 4 and 5
                    if (currentPhase >= 4) {
                        propulsionSystem.children[0].material.emissiveIntensity = 0.3 + Math.sin(time * 3) * 0.3;
                    }
                    
                    // Small reactors pulse in phase 2 and 5
                    if (currentPhase === 2 || currentPhase === 5) {
                        for (let i = 1; i <= 3; i++) {
                            propulsionSystem.children[i].material.emissiveIntensity = 0.3 + Math.sin(time * 5 + i) * 0.5;
                        }
                    }
                    
                    // Ion thrusters pulse in phase 3 and 5
                    if (currentPhase === 3 || currentPhase === 5) {
                        for (let i = 4; i < 4 + 19; i++) {
                            propulsionSystem.children[i].material.emissiveIntensity = 0.5 + Math.sin(time * 8 + i) * 0.5;
                        }
                    }
                }
            }
            
            renderer.render(scene, camera);
        }
        
        // Initialize charts
        let speedChart, reactorChart;
        
        function initCharts() {
            // Speed chart
            const speedCtx = document.getElementById('speed-chart').getContext('2d');
            speedChart = new Chart(speedCtx, {
                type: 'line',
                data: {
                    labels: Array(20).fill(''),
                    datasets: [{
                        label: 'Speed (km/s)',
                        data: Array(20).fill(0),
                        borderColor: '#5D5CDE',
                        backgroundColor: 'rgba(93, 92, 222, 0.1)',
                        tension: 0.3,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    }
                }
            });
            
            // Reactor output chart
            const reactorCtx = document.getElementById('reactor-chart').getContext('2d');
            reactorChart = new Chart(reactorCtx, {
                type: 'line',
                data: {
                    labels: Array(20).fill(''),
                    datasets: [
                        {
                            label: 'Small Reactors',
                            data: Array(20).fill(0),
                            borderColor: '#00ffff',
                            backgroundColor: 'rgba(0, 255, 255, 0.1)',
                            tension: 0.3,
                            fill: true
                        },
                        {
                            label: 'Central Reactor',
                            data: Array(20).fill(0),
                            borderColor: '#5D5CDE',
                            backgroundColor: 'rgba(93, 92, 222, 0.1)',
                            tension: 0.3,
                            fill: true
                        },
                        {
                            label: 'Ion Thrusters',
                            data: Array(20).fill(0),
                            borderColor: '#ff5500',
                            backgroundColor: 'rgba(255, 85, 0, 0.1)',
                            tension: 0.3,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    }
                }
            });
        }
        
        // Update chart data
        function updateCharts() {
            // Update speed chart
            speedChart.data.datasets[0].data.shift();
            speedChart.data.datasets[0].data.push(speed);
            speedChart.update();
            
            // Update reactor chart based on current phase
            let smallReactorOutput = 0;
            let centralReactorOutput = 0;
            let ionThrusterOutput = 0;
            
            switch (currentPhase) {
                case 1: // Initialization
                    smallReactorOutput = 5;
                    centralReactorOutput = 2;
                    ionThrusterOutput = 1;
                    break;
                case 2: // Small Nuclear Reactions
                    smallReactorOutput = 90;
                    centralReactorOutput = 5;
                    ionThrusterOutput = 10;
                    break;
                case 3: // Ion Thrusters
                    smallReactorOutput = 40;
                    centralReactorOutput = 10;
                    ionThrusterOutput = 95;
                    break;
                case 4: // Central Reactor
                    smallReactorOutput = 20;
                    centralReactorOutput = 80;
                    ionThrusterOutput = 50;
                    break;
                case 5: // Light-Speed Push
                    smallReactorOutput = 100;
                    centralReactorOutput = 100;
                    ionThrusterOutput = 100;
                    break;
            }
            
            // Add some randomness
            smallReactorOutput += Math.random() * 5 - 2.5;
            centralReactorOutput += Math.random() * 5 - 2.5;
            ionThrusterOutput += Math.random() * 5 - 2.5;
            
            // Keep within bounds
            smallReactorOutput = Math.max(0, Math.min(100, smallReactorOutput));
            centralReactorOutput = Math.max(0, Math.min(100, centralReactorOutput));
            ionThrusterOutput = Math.max(0, Math.min(100, ionThrusterOutput));
            
            reactorChart.data.datasets[0].data.shift();
            reactorChart.data.datasets[0].data.push(smallReactorOutput);
            
            reactorChart.data.datasets[1].data.shift();
            reactorChart.data.datasets[1].data.push(centralReactorOutput);
            
            reactorChart.data.datasets[2].data.shift();
            reactorChart.data.datasets[2].data.push(ionThrusterOutput);
            
            reactorChart.update();
        }
        
        // Update phase
        function updatePhase(newPhase) {
            currentPhase = newPhase;
            
            // Update phase indicators
            phaseIndicators.forEach((indicator, index) => {
                if (index + 1 === currentPhase) {
                    indicator.classList.add('phase-active');
                } else {
                    indicator.classList.remove('phase-active');
                }
            });
            
            // Update phase text
            phaseText.textContent = `Phase ${currentPhase}: ${phases[currentPhase - 1].name} - ${phases[currentPhase - 1].description}`;
            
            // Update reactor statuses based on phase
            switch (currentPhase) {
                case 1: // Initialization
                    smallReactor.textContent = "Warming up";
                    centralReactor.textContent = "Inactive";
                    ionThrusters.textContent = "Calibrating";
                    coolingSystem.textContent = "Standby";
                    break;
                case 2: // Small Nuclear Reactions
                    smallReactor.textContent = "Active (90%)";
                    centralReactor.textContent = "Inactive";
                    ionThrusters.textContent = "Standby";
                    coolingSystem.textContent = "Active (50%)";
                    break;
                case 3: // Ion Thrusters
                    smallReactor.textContent = "Stable (40%)";
                    centralReactor.textContent = "Warming up";
                    ionThrusters.textContent = "Active (95%)";
                    coolingSystem.textContent = "Active (70%)";
                    break;
                case 4: // Central Reactor
                    smallReactor.textContent = "Standby (20%)";
                    centralReactor.textContent = "Charging (80%)";
                    ionThrusters.textContent = "Active (50%)";
                    coolingSystem.textContent = "Active (90%)";
                    break;
                case 5: // Light-Speed Push
                    smallReactor.textContent = "MAXIMUM (100%)";
                    centralReactor.textContent = "MAXIMUM (100%)";
                    ionThrusters.textContent = "MAXIMUM (100%)";
                    coolingSystem.textContent = "MAXIMUM (100%)";
                    break;
            }
        }
        
        // Main simulation function
        function runSimulation() {
            if (!isSimulationRunning) return;
            
            // Increment time
            timeElapsed += 1;
            
            // Update progress based on current phase
            const progressIncrement = [0.05, 0.1, 0.15, 0.2, 0.5][currentPhase - 1];
            progress += progressIncrement;
            
            // Update speed based on current phase
            let targetSpeed = 0;
            
            switch (currentPhase) {
                case 1: // Initialization
                    targetSpeed = 5; // km/s
                    break;
                case 2: // Small Nuclear Reactions
                    targetSpeed = 50; // km/s
                    break;
                case 3: // Ion Thrusters
                    targetSpeed = 150; // km/s
                    break;
                case 4: // Central Reactor
                    targetSpeed = 500; // km/s
                    break;
                case 5: // Light-Speed Push
                    targetSpeed = 299792; // km/s (speed of light)
                    break;
            }
            
            // Gradually approach target speed
            speed = speed + (targetSpeed - speed) * 0.05;
            
            // Calculate distance (simplified)
            const distanceIncrement = speed / 100; // Convert to millions of km
            distance += distanceIncrement;
            
            // Update temperature based on phase and cooling
            const baseTemp = [310, 600, 800, 1200, 2000][currentPhase - 1];
            const coolingEffect = [5, 15, 25, 40, 60][currentPhase - 1];
            const targetTemp = baseTemp - coolingEffect;
            
            temperature = temperature + (targetTemp - temperature) * 0.1;
            
            // Update UI
            speedValue.textContent = `${Math.round(speed).toLocaleString()} km/s`;
            tempValue.textContent = `${Math.round(temperature)} K`;
            distanceValue.textContent = `${distance.toFixed(1)} million km`;
            timeValue.textContent = `${timeElapsed} days`;
            
            shipProgress.style.left = `${Math.min(100, progress)}%`;
            
            // Change phase if progress threshold reached
            if (progress >= 20 && currentPhase === 1) {
                updatePhase(2);
            } else if (progress >= 40 && currentPhase === 2) {
                updatePhase(3);
            } else if (progress >= 60 && currentPhase === 3) {
                updatePhase(4);
            } else if (progress >= 80 && currentPhase === 4) {
                updatePhase(5);
            }
            
            // Complete simulation if we reach 100%
            if (progress >= 100) {
                phaseText.textContent = "Mission complete! Mars reached in " + timeElapsed + " days!";
                stopSimulation();
            }
            
            // Update charts
            updateCharts();
        }
        
        // Start simulation
        function startSimulation() {
            isSimulationRunning = true;
            startBtn.style.display = 'none';
            resetBtn.style.display = 'block';
            
            // Reset to phase 1
            updatePhase(1);
            
            // Start simulation interval
            simulationInterval = setInterval(runSimulation, 500);
        }
        
        // Stop simulation
        function stopSimulation() {
            isSimulationRunning = false;
            clearInterval(simulationInterval);
            resetBtn.style.display = 'block';
        }
        
        // Reset simulation
        function resetSimulation() {
            // Stop current simulation
            stopSimulation();
            
            // Reset variables
            progress = 0;
            speed = 0;
            temperature = 293;
            distance = 0;
            timeElapsed = 0;
            currentPhase = 1;
            
            // Reset UI
            updatePhase(1);
            shipProgress.style.left = '0%';
            speedValue.textContent = '0 km/s';
            tempValue.textContent = '293 K';
            distanceValue.textContent = '0 million km';
            timeValue.textContent = '0 days';
            
            // Reset charts
            speedChart.data.datasets[0].data = Array(20).fill(0);
            reactorChart.data.datasets[0].data = Array(20).fill(0);
            reactorChart.data.datasets[1].data = Array(20).fill(0);
            reactorChart.data.datasets[2].data = Array(20).fill(0);
            speedChart.update();
            reactorChart.update();
            
            // Show start button
            startBtn.style.display = 'block';
            resetBtn.style.display = 'none';
        }
        
        // Event listeners
        startBtn.addEventListener('click', startSimulation);
        resetBtn.addEventListener('click', resetSimulation);
        
        // Handle window resize
        window.addEventListener('resize', () => {
            if (renderer) {
                camera.aspect = document.getElementById('canvas-container').offsetWidth / document.getElementById('canvas-container').offsetHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(document.getElementById('canvas-container').offsetWidth, document.getElementById('canvas-container').offsetHeight);
            }
        });
        
        // Initialize everything
        window.addEventListener('load', () => {
            initThreeJs();
            initCharts();
            updatePhase(1);
        });
    </script>
</body>
</html>
