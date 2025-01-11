// src/components/UsageLogList.tsx
import React, { useEffect, useState } from 'react';
import api from '../config/axiosConfig';

interface UsageLog {
    id: number;
    toilet_id: number;
    timestamp: string;
}

const UsageLogList: React.FC = () => {
    const [usageLogs, setUsageLogs] = useState<UsageLog[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchUsageLogs = async () => {
            try {
                const response = await api.get('/usagelogs');
                setUsageLogs(response.data);
            } catch (error) {
                setError('Error fetching usage logs');
                console.error('Error fetching usage logs:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchUsageLogs();
    }, []);

    if (loading) return <p>Loading usage logs...</p>;
    if (error) return <p style={{ color: 'red' }}>{error}</p>;

    return (
        <div>
            <h2>Usage Logs</h2>
            <ul>
                {usageLogs.map((log) => (
                    <li key={log.id}>Toilet ID: {log.toilet_id}, Time: {log.timestamp}</li>
                ))}
            </ul>
        </div>
    );
};

export default UsageLogList;
