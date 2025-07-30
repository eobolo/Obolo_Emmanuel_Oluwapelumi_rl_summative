/**
 * SmartDelivery Three.js Visualization
 * 
 * This file contains the main visualization logic for the SmartDelivery
 * environment, including 3D scene management, real-time updates, and
 * interactive features.
 */

class SmartDeliveryVisualization {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.objects = {};
        this.socket = null;
        this.currentData = null;
        this.isInitialized = false;
        
        // Animation
        this.clock = new THREE.Clock();
        this.mixer = null;
        this.animations = [];
        
        // UI state
        this.showWireframe = false;
        this.showStats = false;
        
        this.init();
    }
    
    init() {
        console.log('Initializing SmartDelivery Visualization...');
        
        // Initialize Three.js scene
        this.initScene();
        this.initCamera();
        this.initRenderer();
        this.initControls();
        this.initLights();
        this.initGround();
        
        // Initialize Socket.IO
        this.initSocket();
        
        // Initialize UI controls
        this.initUIControls();
        
        // Start animation loop
        this.animate();
        
        this.isInitialized = true;
        console.log('Visualization initialized successfully');
        
        // Hide loading screen
        document.getElementById('loading').style.display = 'none';
    }
    
    initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x87CEEB); // Sky blue
        this.scene.fog = new THREE.Fog(0x87CEEB, 50, 200);
    }
    
    initCamera() {
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 30, 50);
        this.camera.lookAt(0, 0, 0);
    }
    
    initRenderer() {
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.setPixelRatio(window.devicePixelRatio);
        
        document.getElementById('scene-container').appendChild(this.renderer.domElement);
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    initControls() {
        // Simple orbit controls
        this.controls = {
            rotation: { x: 0, y: 0 },
            target: { x: 0, y: 0, z: 0 },
            distance: 50,
            isMouseDown: false,
            mousePosition: { x: 0, y: 0 }
        };
        
        // Mouse controls
        this.renderer.domElement.addEventListener('mousedown', (e) => {
            this.controls.isMouseDown = true;
            this.controls.mousePosition = { x: e.clientX, y: e.clientY };
        });
        
        this.renderer.domElement.addEventListener('mouseup', () => {
            this.controls.isMouseDown = false;
        });
        
        this.renderer.domElement.addEventListener('mousemove', (e) => {
            if (this.controls.isMouseDown) {
                const deltaX = e.clientX - this.controls.mousePosition.x;
                const deltaY = e.clientY - this.controls.mousePosition.y;
                
                this.controls.rotation.y += deltaX * 0.01;
                this.controls.rotation.x += deltaY * 0.01;
                
                this.controls.mousePosition = { x: e.clientX, y: e.clientY };
            }
        });
        
        // Mouse wheel for zoom
        this.renderer.domElement.addEventListener('wheel', (e) => {
            this.controls.distance += e.deltaY * 0.1;
            this.controls.distance = Math.max(10, Math.min(100, this.controls.distance));
        });
    }
    
    initLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light (sun)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 500;
        directionalLight.shadow.camera.left = -100;
        directionalLight.shadow.camera.right = 100;
        directionalLight.shadow.camera.top = 100;
        directionalLight.shadow.camera.bottom = -100;
        this.scene.add(directionalLight);
        
        // Point light for school
        const pointLight = new THREE.PointLight(0x4CAF50, 0.5, 30);
        pointLight.position.set(0, 20, 0);
        this.scene.add(pointLight);
    }
    
    initGround() {
        // Create ground plane
        const groundGeometry = new THREE.PlaneGeometry(200, 200);
        const groundMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x90EE90,
            transparent: true,
            opacity: 0.8
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);
        
        // Add grid
        const gridHelper = new THREE.GridHelper(200, 20, 0x888888, 0xcccccc);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);
    }
    
    initSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.socket.emit('request_state');
        });
        
        this.socket.on('state_update', (data) => {
            console.log('Received state_update:', data);
            this.updateVisualization(data);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
        });
    }
    
    initUIControls() {
        // Reset camera
        document.getElementById('reset-camera').addEventListener('click', () => {
            this.resetCamera();
        });
        
        // Toggle wireframe
        document.getElementById('toggle-wireframe').addEventListener('click', () => {
            this.toggleWireframe();
        });
        
        // Toggle stats
        document.getElementById('toggle-stats').addEventListener('click', () => {
            this.toggleStats();
        });
        
        // Fullscreen
        document.getElementById('fullscreen').addEventListener('click', () => {
            this.toggleFullscreen();
        });
    }
    
    updateVisualization(data) {
        if (!data || data.error) {
            console.error('Invalid visualization data:', data);
            return;
        }
        
        this.currentData = data;
        
        // Update school
        this.updateSchool(data.school);
        
        // Update textbooks
        this.updateTextbooks(data.textbooks);
        
        // Update guides
        this.updateGuides(data.guides);
        
        // Update truck
        this.updateTruck(data.truck);
        
        // Update UI
        this.updateUI(data.metrics);
    }
    
    updateSchool(schoolData) {
        if (!schoolData) return;
        
        // Remove existing school
        if (this.objects.school) {
            this.scene.remove(this.objects.school);
        }
        
        // Create school building
        const geometry = new THREE.BoxGeometry(
            schoolData.size.width,
            schoolData.size.height,
            schoolData.size.depth
        );
        
        const material = new THREE.MeshLambertMaterial({ 
            color: schoolData.color,
            transparent: true,
            opacity: 0.9
        });
        
        this.objects.school = new THREE.Mesh(geometry, material);
        this.objects.school.position.set(
            schoolData.position.x,
            schoolData.position.y + schoolData.size.height / 2,
            schoolData.position.z
        );
        this.objects.school.castShadow = true;
        this.objects.school.receiveShadow = true;
        
        // Add school label
        this.addLabel(
            this.objects.school,
            `School\n${schoolData.properties.num_students} students\n${schoolData.properties.location_type}`
        );
        
        this.scene.add(this.objects.school);
    }
    
    updateTextbooks(textbooksData) {
        // Remove existing textbooks
        if (this.objects.textbooks) {
            this.objects.textbooks.forEach(obj => this.scene.remove(obj));
        }
        
        this.objects.textbooks = [];
        
        if (!textbooksData) return;
        
        textbooksData.forEach((textbookData, index) => {
            const geometry = new THREE.BoxGeometry(
                textbookData.size.width,
                textbookData.size.height,
                textbookData.size.depth
            );
            
            const material = new THREE.MeshLambertMaterial({ 
                color: textbookData.color,
                transparent: true,
                opacity: 0.8
            });
            
            const textbook = new THREE.Mesh(geometry, material);
            textbook.position.set(
                textbookData.position.x,
                textbookData.position.y,
                textbookData.position.z
            );
            textbook.castShadow = true;
            
            // Add label
            this.addLabel(
                textbook,
                `${textbookData.properties.subject.toUpperCase()}\n${textbookData.properties.count} books\nQuality: ${textbookData.properties.quality}%`
            );
            
            this.objects.textbooks.push(textbook);
            this.scene.add(textbook);
        });
    }
    
    updateGuides(guidesData) {
        // Remove existing guides
        if (this.objects.guides) {
            this.objects.guides.forEach(obj => this.scene.remove(obj));
        }
        
        this.objects.guides = [];
        
        if (!guidesData) return;
        
        guidesData.forEach((guideData, index) => {
            const geometry = new THREE.BoxGeometry(
                guideData.size.width,
                guideData.size.height,
                guideData.size.depth
            );
            
            const material = new THREE.MeshLambertMaterial({ 
                color: guideData.color,
                transparent: true,
                opacity: 0.9
            });
            
            const guide = new THREE.Mesh(geometry, material);
            guide.position.set(
                guideData.position.x,
                guideData.position.y,
                guideData.position.z
            );
            guide.castShadow = true;
            
            // Add label
            this.addLabel(
                guide,
                `Teacher Guide\n${guideData.properties.subject.toUpperCase()}\n${guideData.properties.count} guides`
            );
            
            this.objects.guides.push(guide);
            this.scene.add(guide);
        });
    }
    
    updateTruck(truckData) {
        if (!truckData) return;
        
        // Remove existing truck
        if (this.objects.truck) {
            this.scene.remove(this.objects.truck);
        }
        
        // Create truck
        const geometry = new THREE.BoxGeometry(
            truckData.size.width,
            truckData.size.height,
            truckData.size.depth
        );
        
        const material = new THREE.MeshLambertMaterial({ 
            color: truckData.color,
            transparent: true,
            opacity: 0.9
        });
        
        this.objects.truck = new THREE.Mesh(geometry, material);
        this.objects.truck.position.set(
            truckData.position.x,
            truckData.position.y,
            truckData.position.z
        );
        this.objects.truck.castShadow = true;
        
        // Add truck label
        const successLevels = ['Low', 'Medium', 'High'];
        this.addLabel(
            this.objects.truck,
            `Delivery Truck\nSuccess: ${successLevels[truckData.properties.delivery_success]}\nTime: ${truckData.properties.time_since_delivery} days`
        );
        
        this.scene.add(this.objects.truck);
    }
    
    addLabel(object, text) {
        // Create canvas for text
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 64;
        
        context.fillStyle = 'rgba(0, 0, 0, 0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        context.fillStyle = 'white';
        context.font = '12px Arial';
        context.textAlign = 'center';
        
        const lines = text.split('\n');
        lines.forEach((line, index) => {
            context.fillText(line, canvas.width / 2, 20 + index * 15);
        });
        
        // Create texture and material
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);
        
        sprite.position.set(0, object.geometry.parameters.height + 2, 0);
        object.add(sprite);
    }
    
    updateUI(metrics) {
        if (!metrics) return;
        
        // Update current state
        document.getElementById('num-students').textContent = this.currentData.school?.properties?.num_students || '-';
        document.getElementById('urgency-level').textContent = ['Low', 'Medium', 'High'][metrics.urgency_level] || '-';
        document.getElementById('textbook-ratio').textContent = (metrics.textbook_ratio * 100).toFixed(1) + '%';
        document.getElementById('avg-quality').textContent = Math.round(metrics.average_quality) + '%';
        
        // Update performance metrics
        document.getElementById('grant-usage').textContent = metrics.grant_usage + '%';
        document.getElementById('infrastructure').textContent = metrics.infrastructure_rating + '%';
        document.getElementById('delivery-success').textContent = ['Low', 'Medium', 'High'][metrics.delivery_success] || '-';
        
        // Update progress bars
        document.getElementById('grant-usage-bar').style.width = metrics.grant_usage + '%';
        document.getElementById('infrastructure-bar').style.width = metrics.infrastructure_rating + '%';
        document.getElementById('delivery-success-bar').style.width = (metrics.delivery_success / 2 * 100) + '%';
    }
    
    resetCamera() {
        this.controls.rotation = { x: 0, y: 0 };
        this.controls.distance = 50;
    }
    
    toggleWireframe() {
        this.showWireframe = !this.showWireframe;
        
        // Update all objects
        Object.values(this.objects).forEach(obj => {
            if (Array.isArray(obj)) {
                obj.forEach(mesh => {
                    if (mesh.material) {
                        mesh.material.wireframe = this.showWireframe;
                    }
                });
            } else if (obj.material) {
                obj.material.wireframe = this.showWireframe;
            }
        });
        
        // Update button state
        document.getElementById('toggle-wireframe').classList.toggle('active', this.showWireframe);
    }
    
    toggleStats() {
        this.showStats = !this.showStats;
        document.getElementById('toggle-stats').classList.toggle('active', this.showStats);
    }
    
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        const delta = this.clock.getDelta();
        
        // Update camera position based on controls
        const x = this.controls.distance * Math.sin(this.controls.rotation.y) * Math.cos(this.controls.rotation.x);
        const y = this.controls.distance * Math.sin(this.controls.rotation.x);
        const z = this.controls.distance * Math.cos(this.controls.rotation.y) * Math.cos(this.controls.rotation.x);
        
        this.camera.position.set(x, y, z);
        this.camera.lookAt(
            this.controls.target.x,
            this.controls.target.y,
            this.controls.target.z
        );
        
        // Animate objects
        if (this.objects.truck) {
            this.objects.truck.rotation.y += delta * 0.5;
        }
        
        // Render scene
        this.renderer.render(this.scene, this.camera);
    }
}

// Initialize visualization when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.visualization = new SmartDeliveryVisualization();
});