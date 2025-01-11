// src/components/GenderList.tsx
import React, { useEffect, useState } from 'react';
import api from '../config/axiosConfig';

interface Gender {
    id: number;
    label: string;
}

const GenderList: React.FC = () => {
    const [genders, setGenders] = useState<Gender[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchGenders = async () => {
            try {
                const response = await api.get('/genders');
                setGenders(response.data);
            } catch (error) {
                setError('Error fetching genders');
                console.error('Error fetching genders:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchGenders();
    }, []);

    if (loading) return <p>Loading genders...</p>;
    if (error) return <p style={{ color: 'red' }}>{error}</p>;

    return (
        <div>
            <h2>Genders</h2>
            <ul className="toilet-row">
                {genders.map((gender) => (
                    <li key={gender.id}>{gender.label}</li>
                ))}
            </ul>
        </div>
    );
};

export default GenderList;
