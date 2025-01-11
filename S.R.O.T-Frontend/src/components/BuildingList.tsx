import React, { useEffect, useState } from 'react';
import api from '../config/axiosConfig';

interface BuildingListProps {
    onSelect: (buildingId: number) => void;
}

const BuildingList: React.FC<BuildingListProps> = ({ onSelect }) => {
    const [buildings, setBuildings] = useState<any[]>([]);

    useEffect(() => {
        api.get('/buildings')
            .then(response => setBuildings(response.data))
            .catch(error => console.error('Error fetching buildings:', error));
    }, []);

    return (
        <div className="building-buttons">
            {buildings.map((building) => (
                <button
                    key={building.id}
                    onClick={() => onSelect(building.id)}
                    className="building-button"
                >
                    {building.name}
                </button>
            ))}
        </div>
    );
};

export default BuildingList;


