import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './css/mainpage.css';
import BuildingList from './components/BuildingList';

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

const JanitorMainPage: React.FC = () => {
    const [buildings, setBuildings] = useState<Building[]>([]);
    const [currentBuilding, setCurrentBuilding] = useState<Building | null>(null);
    const [isDropdownOpen, setDropdownOpen] = useState(false);

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
        <div className="main-page">
            <header>
                <nav className="hotbar" style={{ position: 'relative', top: '50px' }}>
                    <div className="nav-item"><a href="/janitor">Toilet</a></div>
                    <div className="nav-item"><a href="/janitorabout-us">About Us</a></div>
                    <div className="nav-item"><a href="/janitordata">Data</a></div>
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
                            <div className="floor-container">
                                {currentBuilding.floors.map((floor) => (
                                    <div key={floor.id} className="floor-box">
                                        <h3>Floor {floor.floor_number}</h3>
                                        <div className="toilet-row">
                                            {floor.toilets.map((toilet) => (
                                                <div key={toilet.id} className="toilet-info">
                                                    <img
                                                        src={(`./gallery/${toilet.gender.label.toLowerCase()}.png`)}
                                                        alt={`${toilet.gender.label} icon`}
                                                        className="gender-icon"
                                                    />
                                                    <span className="center">{toilet.gender.label}<br /></span>
                                                    <pre className="center">Usage<br />Count: {toilet.usage_count}</pre>
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

export default JanitorMainPage;
