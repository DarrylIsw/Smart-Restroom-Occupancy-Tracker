import React, { useEffect, useState } from 'react';
import api from '../config/axiosConfig';

interface Gender {
    label: string;
}

interface Toilet {
    id: number;
    gender: Gender;
    usage_count: number;
}

interface ToiletListProps {
    floorId: number;
}

const ToiletList: React.FC<ToiletListProps> = ({ floorId }) => {
    const [toilets, setToilets] = useState<Toilet[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const fetchToilets = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await api.get(`/floors/${floorId}/toilets`);
            setToilets(response.data);
        } catch (error) {
            setError('Error fetching toilets');
            console.error('Error fetching toilets:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchToilets();
    }, [floorId]);

    const handleResetCounter = async (toiletId: number) => {
        try {
            // Adjusted the API endpoint to include `floorId`
            await api.put(`/floors/${floorId}/toilets/${toiletId}/reset`);
            // Update the UI by re-fetching the toilets
            fetchToilets();
        } catch (error) {
            console.error('Error resetting counter:', error);
            setError('Error resetting counter');
        }
    };

    if (loading) return <p>Loading toilets...</p>;
    if (error) return <p style={{ color: 'red' }}>{error}</p>;

    return (
        <div>
            <h3>Toilets</h3>
            <ul>
                {toilets.length > 0 ? (
                    toilets.map((toilet) => (
                        <li key={toilet.id}>
                            <img
                                src={`http://localhost:8000/gallery/${toilet.gender.label.toLowerCase()}.png`}
                                alt={`${toilet.gender.label} Icon`}
                                className="gender-icon"
                            />
                            <p>Usage Count: {toilet.usage_count}</p>
                            <button onClick={() => handleResetCounter(toilet.id)}>
                                Reset Counter
                            </button>
                        </li>
                    ))
                ) : (
                    <p>No toilets available on this floor.</p>
                )}
            </ul>
        </div>
    );
};

export default ToiletList;
