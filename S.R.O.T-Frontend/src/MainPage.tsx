import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './css/mainpage.css';
import BuildingList from './components/BuildingList';
import FloorList from './components/FloorList';


interface Toilet {
    id: number;
    gender: {
        label: string;
    };
    usage_count: number;
}

interface Floor {
    id: number;
    floor_number: number;
    toilets: Toilet[];
}

interface Building {
    id: number;
    name: string;
    floors: Floor[];
}

const MainPage: React.FC = () => {
    const [buildings, setBuildings] = useState<Building[]>([]);
    const [currentBuilding, setCurrentBuilding] = useState<Building | null>(null);
    const [refreshFloors, setRefreshFloors] = useState(false);
    const [newFloorNumber, setNewFloorNumber] = useState<number | ''>('');

    useEffect(() => {
        const fetchBuildings = async () => {
            try {
                const response = await axios.get('http://localhost:8000/api/buildings');
                setBuildings(response.data);
            } catch (error) {
                console.error('Error fetching buildings:', error);
            }
        };

        fetchBuildings();
    }, []);

    const handleBuildingSelect = async (buildingId: number) => {
        try {
            const response = await axios.get(`http://localhost:8000/api/buildings/${buildingId}/floors`);
            setCurrentBuilding({
                id: buildingId,
                name: `Building ${buildingId}`,
                floors: response.data,
            });
        } catch (error) {
            console.error('Error fetching floors:', error);
        }
    };

    const addNewFloor = async () => {
        if (!currentBuilding || newFloorNumber === '') return;

        try {
            await axios.post(`http://localhost:8000/api/buildings/${currentBuilding.id}/floors`, {
                floor_number: newFloorNumber,
            });
            setRefreshFloors((prev) => !prev);
            setNewFloorNumber('');
        } catch (error) {
            console.error('Error adding new floor:', error);
        }
    };

    const [loadingAction, setLoadingAction] = useState(false);

    const resetUsageCount = async (toiletId: number) => {
        setLoadingAction(true);
        try {
            await axios.put(`http://localhost:8000/api/floors/${currentBuilding?.id}/toilets/${toiletId}/reset`);
            await handleBuildingSelect(currentBuilding?.id!); // Refresh the data
        } catch (error) {
            console.error('Error resetting usage count:', error);
        } finally {
            setLoadingAction(false);
        }
    };
    

    const logToiletUsage = async (toiletId: number) => {
        try {
            await axios.post(`http://localhost:8000/api/toilets/${toiletId}/log`);
            alert('Toilet usage count logged successfully.');
        } catch (error) {
            console.error('Error logging toilet usage:', error);
        }
    };    

    const [isDropdownOpen, setDropdownOpen] = useState(false);
    const [isDeleteDropdownOpen, setDeleteDropdownOpen] = useState(false);
    
    const deleteFloor = async (floorId: number) => {
        if (!currentBuilding) return;
        if (!window.confirm('Are you sure you want to delete this floor? This will also delete associated toilets and logs.')) return;

        try {
            await axios.delete(`http://localhost:8000/api/floors/${floorId}`);
            alert('Floor deleted successfully.');
            await handleBuildingSelect(currentBuilding.id); // Refresh building floors
        } catch (error) {
            console.error('Error deleting floor:', error);
            alert('Failed to delete floor.');
        }
    };

    const handleLogout = async () => {
        try {
            await axios.post(
                'http://localhost:8000/api/logout',
                {}, // No data needed
                { withCredentials: false } // Include credentials
            );
            alert('Logout successful!');
            window.location.href = '/'; // Redirect to login page
        } catch (error) {
            console.error('Logout failed:', error);
            alert('Logout failed. Please try again.');
        }
    };   
    
    return (
        <div className="main-page off">
            <header>
                <nav className="hotbar" style={{ position: 'relative', top: '50px' }}>
                    <div className="nav-item"><a href="/admin0x0x0x0x">Toilet</a></div>
                    <div className="nav-item"><a href="/about-us">About Us</a></div>
                    <div className="nav-item"><a href="/data">Data</a></div>
                </nav>
            </header>
            <body>
                <nav className="tabs">
                        <div className="gedung-container">
                            <BuildingList onSelect={handleBuildingSelect} />
                        </div>
                    <span className="srot">S.R.O.T</span>
                    
                    <div className="dropdown">
                        <img
                            alt="Logo"
                            className="logo"
                            onClick={() => setDropdownOpen(!isDropdownOpen)}
                        />
                        {isDropdownOpen && (
                            <div className="dropdown-content">
                               <button onClick={handleLogout} className="logout-button">Logout</button>
                            </div>
                        )}
                    </div>
                </nav>
                <main id="content" style={{ overflowY: 'auto', maxHeight: '80vh' }}>
                    {currentBuilding ? (
                        <>
                            <h2 className="card-title">{currentBuilding.name}</h2>

                            <details>
                                <summary>Add a new floor</summary>
                                <div>
                                    <div className="add-floor-section">
                                        <input
                                            type="text"
                                            value={newFloorNumber}
                                            onChange={(e) => {
                                                const value = e.target.value;
                                                if (/^\d*$/.test(value)) {
                                                    setNewFloorNumber(Number(value));
                                                }
                                            }}
                                            placeholder="Enter floor number"
                                            className="floor-number-input"
                                        />
                                        <button onClick={addNewFloor} className="add-floor-button">Add Floor</button>
                                    </div>
                                </div>
                            </details>

                            <div className="floor-container">
                                {currentBuilding.floors.map((floor) => (
                                    <div key={floor.id} className="floor-box">
                                        <div className="floor-header" onClick={() => setDeleteDropdownOpen(!isDeleteDropdownOpen)}>
                                            <h3 style={{ fontSize: '20px' }}>
                                                Floor {floor.floor_number}{isDeleteDropdownOpen ? '▸' : '▾'}
                                            </h3>
                                            {isDeleteDropdownOpen && (
                                                <button
                                                    onClick={() => deleteFloor(floor.id)}
                                                    className="delete-floor-button"
                                                >
                                                    Delete Floor
                                                </button>
                                            )}
                                        </div>
                                        <div className="toilet-row">
                                            {floor.toilets.map((toilet) => (
                                                <div key={toilet.id} className="toilet-info">
                                                    <img
                                                        src={(`./gallery/${toilet.gender.label.toLowerCase()}.png`)}
                                                        alt={`${toilet.gender.label} icon`}
                                                        className="gender-icon"
                                                    />
                                                    <span className="center" style={{ fontWeight: 'bold', textTransform: 'uppercase' }}>{toilet.gender.label}<br /></span>
                                                    <pre className="center"> Usage<br />Count: {toilet.usage_count}<br></br></pre>
                                                    <button
                                                        onClick={() => resetUsageCount(toilet.id)}
                                                        className="reset-button"  //THE RESET BUTTON CLASS IS HERE OK ZIN
                                                    >
                                                        Reset Counter
                                                    </button>
                                                    <button
                                                    onClick={() => logToiletUsage(toilet.id)}
                                                    className="log-button" // Add a CSS class for styling
                                                    >
                                                        Log Counter
                                                        </button>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </>
                    ) : (
                        <div className="welcome">
                        <h1>Please select a building to see the details.</h1>
                    </div>
                    )}
                </main>
            </body>
        </div>
    );
};

export default MainPage;













