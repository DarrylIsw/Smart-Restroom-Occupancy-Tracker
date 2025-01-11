import React, { useEffect, useState } from 'react';
import api from '../config/axiosConfig';

interface Floor {
    id: number;
    floor_number: number;
}

interface FloorListProps {
    buildingId: number;
    refreshFloors?: boolean; // Optional prop to trigger a re-fetch of floors
    onSelectFloor?: (floorId: number) => void; // Callback to handle floor selection
}

const FloorList: React.FC<FloorListProps> = ({ buildingId, refreshFloors = false, onSelectFloor }) => {
    const [floors, setFloors] = useState<Floor[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const fetchFloors = async () => {
        try {
            const response = await api.get(`/buildings/${buildingId}/floors`);
            setFloors(response.data);
        } catch (error) {
            setError('Error fetching floors');
            console.error('Error fetching floors:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchFloors();
    }, [buildingId, refreshFloors]); // Re-fetch when buildingId or refreshFloors changes

    if (loading) return <p>Loading floors...</p>;
    if (error) return <p style={{ color: 'red' }}>{error}</p>;

    return (
        <div>
            <h1 style={{ fontSize: '35px' }}>Floors</h1>
            <ul>
                {floors.map((floor) => (
                    <li key={floor.id} onClick={() => onSelectFloor && onSelectFloor(floor.id)}>
                        Floor {floor.floor_number}
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default FloorList;

