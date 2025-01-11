import React, { useEffect, useState } from 'react';
import axios from 'axios';
import * as XLSX from 'xlsx';
import '../css/datapage.css';

interface Log {
    id: number;
    toilet_id: number;
    floor_id: number;
    building_id: number;
    usage_count: number;
    logged_at: string;
    is_reset: boolean;
    toilet: {
        id: number;
        floor_id: number;
        gender_id: number;
        usage_count: number;
    };
    floor: {
        id: number;
        building_id: number;
        floor_number: number;
    };
    building: {
        id: number;
        name: string;
    };
}

const JanitorDataPage: React.FC = () => {
    const [logs, setLogs] = useState<Log[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const response = await axios.get('http://localhost:8000/api/logging-counters');
                setLogs(response.data);
            } catch (error) {
                console.error('Error fetching logs:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchLogs();
    }, []);

    const handleDownloadExcel = () => {
        const formattedData = logs.map((log) => ({
            ID: log.id,
            Toilet_ID: log.toilet_id,
            Building_Name: log.building?.name || 'N/A',
            Floor_Number: log.floor?.floor_number || 'N/A',
            Gender: log.toilet?.gender_id === 1 ? 'Male' : 'Female',
            Usage_Count: log.usage_count,
            Logged_At: new Date(log.logged_at).toLocaleString(),
        }));

        const worksheet = XLSX.utils.json_to_sheet(formattedData);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, 'Logs');

        XLSX.writeFile(workbook, 'Toilet_Usage_Logs.xlsx');
    };

    const handleClearLogs = async () => {
        if (!window.confirm('Are you sure you want to clear all logs? This action cannot be undone.')) {
            return;
        }

        try {
            await axios.delete('http://localhost:8000/api/logging-counters/clear');
            alert('All logs have been cleared successfully.');
            setLogs([]); // Clear logs from the state
        } catch (error) {
            console.error('Error clearing logs:', error);
            alert('Failed to clear logs. Please try again.');
        }
    };

    if (loading) return <p>Loading data...</p>;

    return (
        <div className="main-page">
            <header>
                <nav className="hotbar" style={{ position: 'relative', top: '50px' }}>
                    <div className="nav-item"><a href="/janitor">Toilet</a></div>
                    <div className="nav-item"><a href="/janitorabout-us">About Us</a></div>
                    <div className="nav-item"><a href="/janitordata">Data</a></div>
                </nav>
            </header>
            {/* <div className="data-title">
            <h1>Toilet Usage Data Logs</h1>
            </div> might not use*/}
            <button className="back-button-data" onClick={() => window.history.back()}>
                        ⬅ Back to Main Page
                    </button>
        <div className="data-page">
            <header className="data-page-header">
            <h1>Toilet Usage Data Logs</h1>
            </header>
            <table className="data-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Toilet ID</th>
                        <th>Building Name</th>
                        <th>Floor Number</th>
                        <th>Gender</th>
                        <th>Usage Count</th>
                        <th>Logged At</th>
                        <th>Action Type</th>
                    </tr>
                </thead>
                <tbody>
                    {logs.map((log) => (
                        <tr key={log.id}>
                            <td>{log.id}</td>
                            <td>{log.toilet_id}</td>
                            <td>{log.building?.name || 'N/A'}</td>
                            <td>{log.floor?.floor_number || 'N/A'}</td>
                            <td>{log.toilet?.gender_id === 1 ? 'Male' : 'Female'}</td>
                            <td>{log.usage_count}</td>
                            <td>{new Date(log.logged_at).toLocaleString()}</td>
                            <td>{log.is_reset ? 'Reset' : 'Usage'}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    
    </div>
    );
};

export default JanitorDataPage;