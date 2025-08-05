import React, { useState, useRef, useCallback, useEffect } from 'react';

// Simple icon components to replace lucide-react
const PlayIcon = () => <span>▶️</span>;
const StopIcon = () => <span>⏹️</span>;
const ResetIcon = () => <span>🔄</span>;
const DownloadIcon = () => <span>💾</span>;
const UploadIcon = () => <span>📂</span>;
const SettingsIcon = () => <span>⚙️</span>;

const SystemDynamicsModeler = () => {
const [elements, setElements] = useState([]);
const [connections, setConnections] = useState([]);
const [selectedElement, setSelectedElement] = useState(null);
const [draggedElement, setDraggedElement] = useState(null);
const [isSimulating, setIsSimulating] = useState(false);
const [simulationResults, setSimulationResults] = useState(null);
const [isConnecting, setIsConnecting] = useState(false);
const [connectionStart, setConnectionStart] = useState(null);
const [showSimSettings, setShowSimSettings] = useState(false);
const [showResults, setShowResults] = useState(false);
const [simulationParams, setSimulationParams] = useState({
    startTime: 0,
    endTime: 50,
    timeStep: 1,
    timeUnit: 'years'
  });
  const canvasRef = useRef(null);
  const [nextId, setNextId] = useState(1);

  // Debug logging
  console.log('🔍 Current state:', { 
    simulationResults: simulationResults ? 'HAS_DATA' : 'NULL',
    showResults,
    isSimulating,
    elementsCount: elements.length,
    connectionsCount: connections.length
  });

  // Element types
  const elementTypes = {
    stock: { color: '#3B82F6', icon: '📦', name: 'Stock' },
    flow: { color: '#10B981', icon: '🌊', name: 'Flow' },
    connector: { color: '#F59E0B', icon: '🔗', name: 'Connector' },
    parameter: { color: '#8B5CF6', icon: '⚙️', name: 'Parameter' }
  };

  // Add keyboard support
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'Delete' && selectedElement) {
        deleteElement(selectedElement.id);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [selectedElement, elements, connections]);

  // Handle drag start from palette
  const handleDragStart = (elementType) => {
    setDraggedElement(elementType);
  };

  // Add these new state variables
const [isDraggingElement, setIsDraggingElement] = useState(false);
const [draggedElementId, setDraggedElementId] = useState(null);
const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });




  // Handle drop on canvas
  const handleCanvasDrop = (e) => {
    e.preventDefault();
    if (!draggedElement) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const newElement = {
      id: nextId,
      type: draggedElement,
      x: x - 50, // Center the element
      y: y - 25,
      name: `${elementTypes[draggedElement].name} ${nextId}`,
      value: draggedElement === 'stock' ? 100 : draggedElement === 'parameter' ? 1 : 0,
      formula: draggedElement === 'flow' ? '' : '',
      currentValue: draggedElement === 'stock' ? 100 : undefined,
      currentRate: draggedElement === 'flow' ? 0 : undefined
    };

    setElements([...elements, newElement]);
    setSelectedElement(newElement);
    setNextId(nextId + 1);
    setDraggedElement(null);
  };

// Add these drag handling functions
const handleElementMouseDown = (e, element) => {
  if (isConnecting) return; // Don't drag in connect mode
  
  e.stopPropagation();
  setIsDraggingElement(true);
  setDraggedElementId(element.id);
  
  const canvas = canvasRef.current;
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  setDragOffset({
    x: mouseX - element.x,
    y: mouseY - element.y
  });
};

const handleCanvasMouseMove = (e) => {
  if (!isDraggingElement || !draggedElementId) return;
  
  const canvas = canvasRef.current;
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  const newX = mouseX - dragOffset.x;
  const newY = mouseY - dragOffset.y;
  
  // Update element position
  setElements(elements.map(el => 
    el.id === draggedElementId 
      ? { ...el, x: newX, y: newY }
      : el
  ));
  
  // Update selected element if it's the one being dragged
  if (selectedElement?.id === draggedElementId) {
    setSelectedElement({ ...selectedElement, x: newX, y: newY });
  }
};

const handleCanvasMouseUp = () => {
  setIsDraggingElement(false);
  setDraggedElementId(null);
  setDragOffset({ x: 0, y: 0 });
};


  // Delete element function
  const deleteElement = (elementId) => {
    if (window.confirm('Are you sure you want to delete this element?')) {
      // Remove element
      setElements(elements.filter(el => el.id !== elementId));
      
      // Remove all connections involving this element
      setConnections(connections.filter(conn => 
        conn.fromElement.id !== elementId && conn.toElement.id !== elementId
      ));
      
      // Clear selection if deleted element was selected
      if (selectedElement?.id === elementId) {
        setSelectedElement(null);
      }
    }
  };

  // Handle element selection and connection
  const handleElementClick = (element) => {
    if (isConnecting) {
      if (!connectionStart) {
        // Start connection
        setConnectionStart(element);
      } else {
        // Complete connection
        if (connectionStart.id !== element.id) {
          const newConnection = {
            id: nextId,
            fromElement: connectionStart,
            toElement: element,
            type: determineConnectionType(connectionStart, element)
          };
          setConnections([...connections, newConnection]);
          setNextId(nextId + 1);
        }
        setConnectionStart(null);
        setIsConnecting(false);
      }
    } else {
      setSelectedElement(element);
    }
  };

  // Determine connection type based on elements
  const determineConnectionType = (from, to) => {
    if (from.type === 'stock' && to.type === 'flow') return 'outflow';
    if (from.type === 'flow' && to.type === 'stock') return 'inflow';
    if (from.type === 'parameter' && to.type === 'flow') return 'parameter';
    return 'dependency';
  };

  // Toggle connection mode
  const toggleConnectionMode = () => {
    setIsConnecting(!isConnecting);
    setConnectionStart(null);
  };

  // Delete selected connection
  const deleteConnection = (connectionId) => {
    setConnections(connections.filter(conn => conn.id !== connectionId));
  };

  // Update selected element properties
  const updateElementProperty = (property, value) => {
    if (!selectedElement) return;

    setElements(elements.map(el => 
      el.id === selectedElement.id 
        ? { ...el, [property]: value }
        : el
    ));
    setSelectedElement({ ...selectedElement, [property]: value });
  };

  // Run simulation with enhanced debugging
  const runSimulation = async () => {
    setIsSimulating(true);
    setShowResults(true); 
    
    try {
      const modelData = {
        elements: elements,
        connections: connections.map(conn => ({
          id: conn.id,
          from_element_id: conn.fromElement.id,
          to_element_id: conn.toElement.id,
          connection_type: conn.type
        })),
        simulation_params: {
          start_time: simulationParams.startTime,
          end_time: simulationParams.endTime,
          dt: simulationParams.timeStep
        }
      };

      console.log('🚀 Sending to backend:', modelData);

      const response = await fetch('http://localhost:8000/api/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model: modelData })
      });

      const result = await response.json();
      console.log('📥 Received Simulation data from backend');

      if (response.ok ) { 
        if(result.success)  {
          animateSimulationResults(result);

        // console.log('✅ Simulation successful, setting results...');
        //   setSimulationResults({
        //     time: result.time,
        //     timeUnit: simulationParams.timeUnit,
        //     stocks: result.stocks || {},
        //     flows: result.flows || {}
        //   });
        //   console.log('📊 Results set in state');
        }
         else {
          console.error('❌ Backend returned success=false:', result);
          alert(`Simulation failed: ${result.message || 'Unknown error'}`);
        }
      } else {
        console.error('❌ HTTP error:', response.status, result);
        alert(`HTTP Error ${response.status}: ${result.message || 'Server error'}`);
      }
      
    } catch (error) {
      console.error('💥 Network/Parse error:', error);
      alert(`Network error: ${error.message}`);
    } finally {
      setIsSimulating(false);
    }
  };


// animateSimulationResults function

const animateSimulationResults = (result) => {
  const timePoints = result.time.length;
  const animationDuration = 3000; // 3 seconds total animation
  const intervalTime = animationDuration / timePoints;
  
  let currentStep = 0;
  
  const animate = () => {
    if (currentStep >= timePoints) {
      // Animation complete - keep final values displayed
      console.log('✅ Real-time simulation completed - keeping final values');
      
      // Set final simulation results
      setSimulationResults({
        time: result.time,
        timeUnit: simulationParams.timeUnit,
        stocks: result.stocks || {},
        flows: result.flows || {}
      });
      
      // Keep the final values on elements (don't reset)
      setElements(prevElements => 
        prevElements.map(element => {
          if (element.type === 'stock' && result.stocks[element.name]) {
            const finalValue = result.stocks[element.name][result.stocks[element.name].length - 1];
            return {
              ...element,
              currentValue: Math.round(finalValue * 100) / 100
            };
          } else if (element.type === 'flow' && result.flows[element.name]) {
            const finalRate = result.flows[element.name][result.flows[element.name].length - 1];
            return {
              ...element,
              currentRate: Math.round(finalRate * 100) / 100
            };
          }
          return element;
        })
      );
      
      setIsSimulating(false);
      return;
    }
    
    // Update current values on elements during animation
    setElements(prevElements => 
      prevElements.map(element => {
        if (element.type === 'stock' && result.stocks[element.name]) {
          return {
            ...element,
            currentValue: Math.round(result.stocks[element.name][currentStep] * 100) / 100
          };
        } else if (element.type === 'flow' && result.flows[element.name]) {
          return {
            ...element,
            currentRate: Math.round(result.flows[element.name][currentStep] * 100) / 100
          };
        }
        return element;
      })
    );
    
    // Store partial results for the results panel
    const partialResults = {
      time: result.time.slice(0, currentStep + 1),
      timeUnit: simulationParams.timeUnit,
      stocks: {},
      flows: {}
    };
    
    // Get partial data for each stock/flow
    Object.keys(result.stocks || {}).forEach(stockName => {
      partialResults.stocks[stockName] = result.stocks[stockName].slice(0, currentStep + 1);
    });
    
    Object.keys(result.flows || {}).forEach(flowName => {
      partialResults.flows[flowName] = result.flows[flowName].slice(0, currentStep + 1);
    });
    
    setSimulationResults(partialResults);
    
    currentStep++;
    setTimeout(animate, intervalTime);
  };
  
  animate();
};

  // Test function with mock data
  const testWithMockData = () => {
    console.log('🧪 Setting mock results...');
    setSimulationResults({
      time: [0, 1, 2, 3, 4, 5],
      timeUnit: 'years',
      stocks: {
        'Test Stock': [1000, 1020, 1040, 1061, 1082, 1104]
      },
      flows: {
        'Test Flow': [20, 20.4, 20.8, 21.2, 21.6, 22.0]
      }
    });
    setShowResults(true);
    console.log('✅ Mock results set');
  };

// importing models
const importModel = () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json';
  
  input.onchange = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const modelData = JSON.parse(e.target.result);
        
        // Validate the imported data
        if (!modelData.elements || !Array.isArray(modelData.elements)) {
          alert('Invalid model file: Missing elements array');
          return;
        }
        
        if (!modelData.connections || !Array.isArray(modelData.connections)) {
          alert('Invalid model file: Missing connections array');
          return;
        }
        
        // Clear current model
        setElements([]);
        setConnections([]);
        setSelectedElement(null);
        setSimulationResults(null);
        setShowResults(false);
        
        // Load the imported model
        setElements(modelData.elements.map(el => ({
          ...el,
          currentValue: el.type === 'stock' ? el.value : undefined,
          currentRate: el.type === 'flow' ? 0 : undefined
        })));
        setConnections(modelData.connections);
        
        // Update next ID to avoid conflicts
        const maxId = Math.max(
          ...modelData.elements.map(el => el.id),
          ...modelData.connections.map(conn => conn.id),
          0
        );
        setNextId(maxId + 1);
        
        console.log('📂 Model imported successfully:', modelData.metadata?.created || 'Unknown date');
        alert(`Model imported successfully!\nElements: ${modelData.elements.length}\nConnections: ${modelData.connections.length}`);
        
      } catch (error) {
        console.error('Import error:', error);
        alert('Failed to import model: Invalid JSON file');
      }
    };
    
    reader.readAsText(file);
  };
  
  input.click();
};


  // Export model

const exportModel = () => {
  const model = {
    elements: elements.map(el => ({
      ...el,
      // Don't export current runtime values, only original values
      currentValue: undefined,
      currentRate: undefined
    })),
    connections,
    simulationParams, // Include simulation settings
    metadata: {
      created: new Date().toISOString(),
      version: '1.0',
      elementsCount: elements.length,
      connectionsCount: connections.length
    }
  };
  
  const dataStr = JSON.stringify(model, null, 2);
  const dataBlob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(dataBlob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `system_dynamics_model_${new Date().toISOString().split('T')[0]}.json`;
  link.click();
  
  console.log('💾 Model exported successfully');
  
  // Show temporary success message
  const originalElements = elements.length;
  const originalConnections = connections.length;
  alert(`Model exported successfully!\nElements: ${originalElements}\nConnections: ${originalConnections}`);
};

  const styles = {
    container: {
      width: '100%',
      height: '100vh',
      display: 'flex',
      backgroundColor: '#f3f4f6',
      fontFamily: 'Arial, sans-serif'
    },
    sidebar: {
      width: '250px',
      backgroundColor: 'white',
      boxShadow: '2px 0 4px rgba(0,0,0,0.1)',
      padding: '16px'
    },
    componentItem: {
      display: 'flex',
      alignItems: 'center',
      padding: '12px',
      marginBottom: '8px',
      backgroundColor: '#f9fafb',
      borderRadius: '8px',
      cursor: 'grab',
      border: '1px solid #e5e7eb'
    },
    componentIcon: {
      fontSize: '24px',
      marginRight: '12px'
    },
    componentText: {
      flex: 1
    },
    componentName: {
      fontWeight: 'bold',
      marginBottom: '4px'
    },
    componentDesc: {
      fontSize: '12px',
      color: '#6b7280'
    },
    button: {
      width: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      padding: '8px 16px',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer',
      marginBottom: '8px',
      fontSize: '14px',
      fontWeight: 'bold'
    },
    buttonGreen: {
      backgroundColor: '#10b981',
      color: 'white'
    },
    buttonRed: {
      backgroundColor: '#ef4444',
      color: 'white'
    },
    buttonBlue: {
      backgroundColor: '#3b82f6',
      color: 'white'
    },
    buttonPurple: {
      backgroundColor: '#8b5cf6',
      color: 'white'
    },
    buttonGray: {
      backgroundColor: '#6b7280',
      color: 'white'
    },
    mainArea: {
      flex: 1,
      display: 'flex',
      flexDirection: 'column'
    },
    toolbar: {
      height: '48px',
      backgroundColor: 'white',
      borderBottom: '1px solid #e5e7eb',
      display: 'flex',
      alignItems: 'center',
      padding: '0 16px',
      boxShadow: '0 1px 2px rgba(0,0,0,0.1)'
    },
    canvas: {
      flex: 1,
      position: 'relative',
      overflow: 'hidden'
    },
    propertiesPanel: {
      width: '300px',
      backgroundColor: 'white',
      boxShadow: '-2px 0 4px rgba(0,0,0,0.1)',
      padding: '16px'
    },
    input: {
      width: '100%',
      padding: '8px 12px',
      border: '1px solid #d1d5db',
      borderRadius: '4px',
      marginBottom: '12px'
    },
    textarea: {
      width: '100%',
      padding: '8px 12px',
      border: '1px solid #d1d5db',
      borderRadius: '4px',
      height: '80px',
      resize: 'none',
      fontFamily: 'monospace'
    },
    buttonDisabled: {
      opacity: 0.5,
      cursor: 'not-allowed'
    },
    buttonActive: {
      backgroundColor: '#059669',
      boxShadow: '0 0 0 2px #10b981'
    },
    modal: {
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    },
    modalContent: {
      backgroundColor: 'white',
      padding: '24px',
      borderRadius: '8px',
      width: '400px',
      maxWidth: '90%',
      maxHeight: '80%',
      overflow: 'auto'
    },
    resultsPanel: {
      position: 'fixed',
      right: '320px',
      top: '48px',
      bottom: 0,
      width: '400px',
      backgroundColor: 'white',
      boxShadow: '-2px 0 4px rgba(0,0,0,0.1)',
      padding: '16px',
      overflow: 'auto',
      zIndex: 100
    },
    debugPanel: {
      position: 'fixed',
      top: '10px',
      right: '10px',
      backgroundColor: 'yellow',
      padding: '10px',
      border: '2px solid red',
      zIndex: 9999,
      fontSize: '12px',
      maxWidth: '300px',
      borderRadius: '4px'
    }
  };

  return (
    <div style={styles.container}>
      {/* DEBUG PANEL */}
      <div style={styles.debugPanel}>
        <div><strong>DEBUG INFO:</strong></div>
        <div>simulationResults: {simulationResults ? 'EXISTS' : 'NULL'}</div>
        <div>showResults: {showResults.toString()}</div>
        <div>isSimulating: {isSimulating.toString()}</div>
        {simulationResults && (
          <div>
            <div>Time points: {simulationResults.time?.length || 0}</div>
            <div>Stocks: {Object.keys(simulationResults.stocks || {}).length}</div>
            <div>Flows: {Object.keys(simulationResults.flows || {}).length}</div>
          </div>
        )}
      </div>

      {/* Sidebar - Element Palette */}
      <div style={styles.sidebar}>
        <h2 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '16px' }}>Components</h2>
        
        {Object.entries(elementTypes).map(([type, config]) => (
          <div
            key={type}
            draggable
            onDragStart={() => handleDragStart(type)}
            style={styles.componentItem}
          >
            <span style={styles.componentIcon}>{config.icon}</span>
            <div style={styles.componentText}>
              <div style={styles.componentName}>{config.name}</div>
              <div style={styles.componentDesc}>
                {type === 'stock' && 'State variable'}
                {type === 'flow' && 'Rate of change'}
                {type === 'connector' && 'Dependency link'}
                {type === 'parameter' && 'Constant value'}
              </div>
            </div>
          </div>
        ))}

        <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid #e5e7eb' }}>
          <h3 style={{ fontWeight: 'bold', marginBottom: '12px' }}>Connections</h3>
          <button
            onClick={toggleConnectionMode}
            style={{
              ...styles.button, 
              ...(isConnecting ? styles.buttonActive : styles.buttonBlue)
            }}
          >
            🔗 {isConnecting ? 'Connecting...' : 'Connect Elements'}
          </button>
          {isConnecting && (
            <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '8px', padding: '8px', backgroundColor: '#f3f4f6', borderRadius: '4px' }}>
              Click first element, then click second element to connect them
            </div>
          )}
        </div>

        <div style={{ marginTop: '24px' }}>
          <h3 style={{ fontWeight: 'bold', marginBottom: '12px' }}>Simulation</h3>
          <div>
            <button
              onClick={() => setShowSimSettings(true)}
              style={{...styles.button, ...styles.buttonPurple}}
            >
              ⚙️ Settings
            </button>
            
            <button
              onClick={runSimulation}
              disabled={isSimulating}
              style={{
                ...styles.button, 
                ...styles.buttonGreen,
                ...(isSimulating ? styles.buttonDisabled : {})
              }}
            >
              <PlayIcon />
              {isSimulating ? 'Running...' : 'Run'}
            </button>
            
            <button style={{...styles.button, ...styles.buttonRed}}>
              <StopIcon />
              Stop
            </button>
            
            <button 
              onClick={() => {
                setSimulationResults(null);
                setShowResults(false);
                setElements(elements.map(element => ({
                  ...element,
                  currentValue: element.type === 'stock' ? element.value : undefined,
                  currentRate: element.type === 'flow' ? 0 : undefined
                })));

                console.log('🔄 Model reset to initial state');
              }}
              style={{...styles.button, ...styles.buttonBlue}}
            >
              <ResetIcon />
              Reset
            </button>
          </div>
        </div>

        <div style={{ marginTop: '24px' }}>
          <h3 style={{ fontWeight: 'bold', marginBottom: '12px' }}>Model</h3>
          <div>
            <button
              onClick={exportModel}
              style={{...styles.button, ...styles.buttonPurple}}
            >
              <DownloadIcon />
              Export
            </button>
            
            <button 
             onClick={importModel}
            style={{...styles.button, ...styles.buttonGray}}>
              <UploadIcon />
              Import
            </button>
          </div>
        </div>

        {/* Debug section */}
        <div style={{ marginTop: '24px' }}>
          <h3 style={{ fontWeight: 'bold', marginBottom: '12px' }}>Debug</h3>
          <button
            onClick={testWithMockData}
            style={{...styles.button, ...styles.buttonPurple}}
          >
            🧪 Test Results
          </button>
        </div>
      </div>

      {/* Main Canvas Area */}
      <div style={styles.mainArea}>
        {/* Toolbar */}
        <div style={styles.toolbar}>
          <h1 style={{ fontSize: '20px', fontWeight: 'bold' }}>System Dynamics Modeler</h1>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px' }}>
            {simulationResults && (
              <button
                onClick={() => setShowResults(!showResults)}
                style={{
                  ...styles.button,
                  ...styles.buttonBlue,
                  width: 'auto',
                  marginBottom: 0
                }}
              >
                📊 {showResults ? 'Hide Results' : 'Show Results'}
              </button>
            )}
            <div style={{ fontSize: '14px', color: '#6b7280', display: 'flex', alignItems: 'center' }}>
              Elements: {elements.length} | Connections: {connections.length}
            </div>
          </div>
        </div>

        {/* Canvas */}
        <div style={styles.canvas}>
          <svg
            ref={canvasRef}
            style={{
              width: '100%',
              height: '100%',
              backgroundColor: '#f9fafb',
              border: '1px solid #e5e7eb'
            }}
          
          onDrop={handleCanvasDrop}
          onDragOver={(e) => e.preventDefault()}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          >
            {/* Grid pattern */}
            <defs>
              <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e5e7eb" strokeWidth="0.5"/>
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />

            {/* Render connections */}
            {connections.map((conn) => {
              const fromX = conn.fromElement.x + 50;
              const fromY = conn.fromElement.y + 25;
              const toX = conn.toElement.x + 50;
              const toY = conn.toElement.y + 25;
              
              const connectionColor = {
                'inflow': '#10b981',
                'outflow': '#ef4444', 
                'parameter': '#8b5cf6',
                'dependency': '#6b7280'
              }[conn.type] || '#6b7280';

              return (
                <g key={conn.id}>
                  <line
                    x1={fromX}
                    y1={fromY}
                    x2={toX}
                    y2={toY}
                    stroke={connectionColor}
                    strokeWidth="3"
                    markerEnd={`url(#arrowhead-${conn.type})`}
                    style={{ cursor: 'pointer' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm('Delete this connection?')) {
                        deleteConnection(conn.id);
                      }
                    }}
                  />
                  {/* Connection label */}
                  <text
                    x={(fromX + toX) / 2}
                    y={(fromY + toY) / 2 - 5}
                    textAnchor="middle"
                    fill={connectionColor}
                    fontSize="10"
                    fontWeight="bold"
                  >
                    {conn.type}
                  </text>
                </g>
              );
            })}

            {/* Arrow markers for different connection types */}
            <defs>
              {['inflow', 'outflow', 'parameter', 'dependency'].map(type => {
                const color = {
                  'inflow': '#10b981',
                  'outflow': '#ef4444', 
                  'parameter': '#8b5cf6',
                  'dependency': '#6b7280'
                }[type];
                
                return (
                  <marker 
                    key={type}
                    id={`arrowhead-${type}`} 
                    markerWidth="10" 
                    markerHeight="7" 
                    refX="9" 
                    refY="3.5" 
                    orient="auto"
                  >
                    <polygon points="0 0, 10 3.5, 0 7" fill={color} />
                  </marker>
                );
              })}
            </defs>

            {/* Render elements */}
            {elements.map((element) => (
              <g
                key={element.id}
                transform={`translate(${element.x}, ${element.y})`}
                style={{ cursor: isDraggingElement ? 'grabbing' : 'grab' }}
                // style={{ cursor: 'pointer' }}
              >
                <rect
                  width="100"
                  height="50"
                  rx="5"
                  fill={elementTypes[element.type].color}
                  stroke={selectedElement?.id === element.id ? "#000" : "transparent"}
                  strokeWidth="2"
                  opacity="0.8"
                  onMouseDown={(e) => handleElementMouseDown(e, element)}
                  onClick={(e) => {
                    if (!isDraggingElement) {
                      handleElementClick(element);}}}               

                />
                <text
                  x="50"
                  y="20"
                  textAnchor="middle"
                  fill="white"
                  fontSize="12"
                  fontWeight="bold"
                >
                  {element.name}
                </text>
                <text
                  x="50"
                  y="35"
                  textAnchor="middle"
                  fill="white"
                  fontSize="10"
                >
                {element.type === 'stock' ? `Value: ${element.currentValue || element.value}` : 
                element.type === 'parameter' ? `Param: ${element.value}` :
                element.type === 'flow' ? `Rate: ${element.currentRate || 0}` : 'Connector'}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>

      {/* Properties Panel */}
      <div style={styles.propertiesPanel}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
          <span>⚙️</span>
          <h2 style={{ fontSize: '18px', fontWeight: 'bold' }}>Properties</h2>
        </div>

        {selectedElement ? (
          <div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>Name</label>
              <input
                type="text"
                value={selectedElement.name}
                onChange={(e) => updateElementProperty('name', e.target.value)}
                style={styles.input}
              />
            </div>
            
            <div style={{ marginBottom: '16px' }}>
              <button
                onClick={() => deleteElement(selectedElement.id)}
                style={{
                  ...styles.button, 
                  ...styles.buttonRed,
                  fontSize: '12px',
                  padding: '6px 12px'
                }}
              >
                🗑️ Delete Element
              </button>
            </div>

            {selectedElement.type === 'stock' && (
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>Initial Value</label>
                <input
                  type="number"
                  value={selectedElement.value}
                  onChange={(e) => updateElementProperty('value', parseFloat(e.target.value))}
                  style={styles.input}
                />
              </div>
            )}

            {selectedElement.type === 'flow' && (
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>Rate Formula</label>
                <textarea
                  value={selectedElement.formula}
                  onChange={(e) => updateElementProperty('formula', e.target.value)}
                  placeholder="e.g., 0.02 * Population"
                  style={styles.textarea}
                />
                <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px' }}>
                  Use stock names and mathematical operators
                </div>
              </div>
            )}

            {selectedElement.type === 'parameter' && (
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>Value</label>
                <input
                  type="number"
                  step="any"
                  value={selectedElement.value}
                  onChange={(e) => updateElementProperty('value', parseFloat(e.target.value))}
                  style={styles.input}
                />
              </div>
            )}

            {/* Show connections for this element */}
            <div style={{ marginBottom: '16px' }}>
              <h4 style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '8px' }}>Connections</h4>
              {connections.filter(conn => 
                conn.fromElement.id === selectedElement.id || conn.toElement.id === selectedElement.id
              ).length === 0 ? (
                <div style={{ fontSize: '12px', color: '#6b7280' }}>No connections</div>
              ) : (
                <div>
                  {connections.filter(conn => 
                    conn.fromElement.id === selectedElement.id || conn.toElement.id === selectedElement.id
                  ).map(conn => (
                    <div key={conn.id} style={{ 
                      fontSize: '12px', 
                      padding: '4px 8px', 
                      backgroundColor: '#f3f4f6', 
                      borderRadius: '4px', 
                      marginBottom: '4px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}>
                      <span>
                        {conn.fromElement.name} → {conn.toElement.name} ({conn.type})
                      </span>
                      <button
                        onClick={() => deleteConnection(conn.id)}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: '#ef4444',
                          cursor: 'pointer',
                          fontSize: '12px'
                        }}
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div style={{ paddingTop: '16px', borderTop: '1px solid #e5e7eb' }}>
              <div style={{ fontSize: '14px', color: '#6b7280' }}>
                <div>Type: {elementTypes[selectedElement.type].name}</div>
                <div>ID: {selectedElement.id}</div>
                <div>Position: ({selectedElement.x}, {selectedElement.y})</div>
              </div>
            </div>
          </div>
        ) : (
          <div style={{ color: '#6b7280', textAlign: 'center', padding: '32px 0' }}>
            Select an element to view its properties
          </div>
        )}

        {/* Simulation Status */}
        {/* {isSimulating && (
          <div style={{ marginTop: '16px', padding: '12px', backgroundColor: '#fef3c7', borderRadius: '4px', fontSize: '12px' }}>
            <div style={{ color: '#92400e' }}>🔄 Running simulation on Python backend...</div>
          </div>
        )} */}

        {/* Add this after the existing simulation status */}
        {isSimulating && (
          <div style={{ marginTop: '16px', padding: '12px', backgroundColor: '#dbeafe', borderRadius: '4px', fontSize: '12px' }}>
            <div style={{ color: '#1d4ed8', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div className="animate-spin">🔄</div>
              <div>
                <div>Running live simulation...</div>
                <div style={{ fontSize: '10px', marginTop: '4px' }}>
                  Watch values update in real-time on canvas!
                </div>
              </div>
            </div>
          </div>
        )}


        {/* Simulation Results Summary */}
        {simulationResults && (
          <div style={{ marginTop: '32px', paddingTop: '16px', borderTop: '1px solid #e5e7eb' }}>
            <h3 style={{ fontWeight: 'bold', marginBottom: '12px' }}>Simulation Results</h3>
            <div style={{ backgroundColor: '#f9fafb', padding: '12px', borderRadius: '4px', fontSize: '12px' }}>
              <div>Time: {simulationResults.time[0]} - {simulationResults.time[simulationResults.time.length - 1]} {simulationResults.timeUnit}</div>
              <div>Time steps: {simulationResults.time.length}</div>
              <div>Stocks tracked: {Object.keys(simulationResults.stocks).length}</div>
              <div>Flows tracked: {Object.keys(simulationResults.flows).length}</div>
              <div style={{ marginTop: '8px', color: '#10b981' }}>✓ Simulation completed</div>
            </div>
          </div>
        )}
      </div>

      {/* Simulation Settings Modal */}
      {showSimSettings && (
        <div style={styles.modal} onClick={() => setShowSimSettings(false)}>
          <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '16px' }}>Simulation Settings</h2>
            
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>Start Time</label>
              <input
                type="number"
                value={simulationParams.startTime}
                onChange={(e) => setSimulationParams({...simulationParams, startTime: parseFloat(e.target.value)})}
                style={styles.input}
              />
            </div>

            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>End Time</label>
              <input
                type="number"
                value={simulationParams.endTime}
                onChange={(e) => setSimulationParams({...simulationParams, endTime: parseFloat(e.target.value)})}
                style={styles.input}
              />
            </div>

            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>Time Step</label>
              <input
                type="number"
                step="0.1"
                value={simulationParams.timeStep}
                onChange={(e) => setSimulationParams({...simulationParams, timeStep: parseFloat(e.target.value)})}
                style={styles.input}
              />
            </div>

            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>Time Unit</label>
              <select
                value={simulationParams.timeUnit}
                onChange={(e) => setSimulationParams({...simulationParams, timeUnit: e.target.value})}
                style={styles.input}
              >
                <option value="seconds">Seconds</option>
                <option value="minutes">Minutes</option>
                <option value="hours">Hours</option>
                <option value="days">Days</option>
                <option value="weeks">Weeks</option>
                <option value="months">Months</option>
                <option value="years">Years</option>
              </select>
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => setShowSimSettings(false)}
                style={{...styles.button, ...styles.buttonGreen, flex: 1}}
              >
                Save Settings
              </button>
              <button
                onClick={() => setShowSimSettings(false)}
                style={{...styles.button, ...styles.buttonGray, flex: 1}}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Results Panel */}
      {showResults && simulationResults && (
        <div style={styles.resultsPanel}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: 'bold' }}>📊 Results</h2>
            <button
              onClick={() => setShowResults(false)}
              style={{ background: 'none', border: 'none', fontSize: '18px', cursor: 'pointer' }}
            >
              ✕
            </button>
          </div>

          <div style={{ marginBottom: '24px' }}>
            <h3 style={{ fontWeight: 'bold', marginBottom: '8px' }}>Stock Values Over Time</h3>
            {Object.entries(simulationResults.stocks).map(([stockName, values]) => (
              <div key={stockName} style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>{stockName}</div>
                <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '8px' }}>
                  Initial: {values[0]?.toFixed(2)} | Final: {values[values.length - 1]?.toFixed(2)}
                </div>
                <div style={{ 
                  height: '60px', 
                  backgroundColor: '#f3f4f6', 
                  borderRadius: '4px', 
                  padding: '8px',
                  position: 'relative',
                  overflow: 'hidden'
                }}>
                  {/* Simple mini chart */}
                  <svg width="100%" height="44">
                    <polyline
                      points={values.map((value, index) => 
                        `${(index / (values.length - 1)) * 100}%,${44 - ((value - Math.min(...values)) / (Math.max(...values) - Math.min(...values))) * 44}`
                      ).join(' ')}
                      fill="none"
                      stroke="#3b82f6"
                      strokeWidth="2"
                    />
                  </svg>
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginBottom: '24px' }}>
            <h3 style={{ fontWeight: 'bold', marginBottom: '8px' }}>Flow Rates Over Time</h3>
            {Object.entries(simulationResults.flows).map(([flowName, values]) => (
              <div key={flowName} style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>{flowName}</div>
                <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '8px' }}>
                  Avg: {(values.reduce((a, b) => a + b, 0) / values.length).toFixed(2)}
                </div>
                <div style={{ 
                  height: '60px', 
                  backgroundColor: '#f3f4f6', 
                  borderRadius: '4px', 
                  padding: '8px',
                  position: 'relative',
                  overflow: 'hidden'
                }}>
                  <svg width="100%" height="44">
                    <polyline
                      points={values.map((value, index) => 
                        `${(index / (values.length - 1)) * 100}%,${44 - ((value - Math.min(...values)) / (Math.max(...values) - Math.min(...values))) * 44}`
                      ).join(' ')}
                      fill="none"
                      stroke="#10b981"
                      strokeWidth="2"
                    />
                  </svg>
                </div>
              </div>
            ))}
          </div>

          <div style={{ fontSize: '12px', color: '#6b7280', textAlign: 'center' }}>
            Simulation: {simulationParams.startTime} - {simulationParams.endTime} {simulationParams.timeUnit}
          </div>
        </div>
      )}
    </div>
  );
};

export default SystemDynamicsModeler;