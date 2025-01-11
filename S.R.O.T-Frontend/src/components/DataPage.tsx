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

const DataPage: React.FC = () => {
    const [logs, setLogs] = useState<Log[]>([]);
    const [loading, setLoading] = useState(true);

    const [filteredLogs, setFilteredLogs] = useState<Log[]>([]);
    const [showFilter, setShowFilter] = useState(false);
    const [filter, setFilter] = useState({
        buildingId: '',
        toiletId: '',
        startTime: '',
        endTime: '',
        genderId: '',
        startDate: '',
        endDate: '',
        floorId: '',
    });

    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const response = await axios.get('http://localhost:8000/api/logging-counters');
                setLogs(response.data);
                setFilteredLogs(response.data); // Set the initial filtered logs
            } catch (error) {
                console.error('Error fetching logs:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchLogs();
    }, []);

    const handleDownloadExcel = () => {
        const formattedData = filteredLogs.map((log) => ({
            ID: log.id,
            // Toilet_ID: log.toilet_id,
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
            setFilteredLogs([]); // Clear filtered logs as well
        } catch (error) {
            console.error('Error clearing logs:', error);
            alert('Failed to clear logs. Please try again.');
        }
    };

    const handleFilterChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFilter((prev) => ({ ...prev, [name]: value }));
    };
    

    const applyFilter = () => {
        const { buildingId, toiletId, genderId, floorId, startDate, startTime, endDate, endTime } = filter;
    
        const filtered = logs.filter((log) => {
            const logDateTime = new Date(log.logged_at);
            const logDate = logDateTime.toISOString().split('T')[0]; // Extract the date part
            const logTime = logDateTime.toTimeString().split(' ')[0]; // Extract the time part (HH:MM:SS)
    
            const isDateInRange =
                (!startDate || logDate >= startDate) &&
                (!endDate || logDate <= endDate);
    
            const isTimeInRange =
                (!startTime || logTime >= startTime) &&
                (!endTime || logTime <= endTime);
    
            return (
                (!buildingId || log.building_id.toString() === buildingId) &&
                (!toiletId || log.toilet_id.toString() === toiletId) &&
                (!genderId || log.toilet.gender_id.toString() === genderId) &&
                (!floorId || log.floor_id.toString() === floorId) &&
                isDateInRange &&
                isTimeInRange
            );
        });
    
        setFilteredLogs(filtered);
    };
    
    

    if (loading) return <p>Loading data...</p>;
    return (
        <div className="main-page off">
            <header>
                <nav className="hotbar" style={{ position: 'relative', top: '50px' }}>
                    <div className="nav-item"><a href="/admin0x0x0x0x">Toilet</a></div>
                    <div className="nav-item"><a href="/about-us">About Us</a></div>
                    <div className="nav-item"><a href="/data">Data</a></div>
                </nav>
            </header>
            <button className="back-button-data" onClick={() => window.history.back()}>
                ⬅ Back to Main Page
            </button>
            <div className="data-page">
                <header className="data-page-header">
                    <h1>Toilet Usage Data Logs</h1>
                    <div className="header-buttons">
                        <button className="download-button" onClick={handleDownloadExcel} title="Download in .xlsx">
                            ⬇ Excel
                        </button>
                        <button className="clear-logs-button" onClick={handleClearLogs} title="Clears the log below">
                            🗑 Clear Logs
                        </button>
                    </div>
                </header>
                <div className="filter-section">
    <h3
        onClick={() => setShowFilter(!showFilter)}
        style={{ cursor: 'pointer', fontWeight: 'bold' }}
        onMouseEnter={(e) => (e.currentTarget.style.textDecoration = 'underline')}
        onMouseLeave={(e) => (e.currentTarget.style.textDecoration = 'none')}
    >
        Filter Logs {showFilter ? '▸' : '▾'}
    </h3>
    {showFilter && (
        <div className="flex w-72 flex-col gap-6">
            <br />
            <select
                name="buildingId"
                value={filter.buildingId}
                onChange={handleFilterChange}
                style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
            >
                <option value="">All Buildings</option>
                {Array.from(new Set(logs.map((log) => log.building_id))).map((id) => (
                    <option key={id} value={id}>
                        Building {id}
                    </option>
                ))}
            </select>
            <select
                name="genderId"
                value={filter.genderId}
                onChange={handleFilterChange}
                style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
            >
                <option value="">All Genders</option>
                <option value="1">Male</option>
                <option value="2">Female</option>
            </select>
            <select
                name="floorId"
                value={filter.floorId}
                onChange={handleFilterChange}
                style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
            >
                <option value="">All Floors</option>
                {Array.from(new Set(logs.map((log) => log.floor_id))).map((id) => (
                    <option key={id} value={id}>
                        Floor {id}
                    </option>
                ))}
            </select>
            <br />
            <br />
            <label><b>Entry Date:</b></label>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                <input
                    type="date"
                    name="startDate"
                    value={filter.startDate}
                    onChange={handleFilterChange}
                    style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
                />
                <input
                    type="time"
                    name="startTime"
                    value={filter.startTime}
                    onChange={handleFilterChange}
                    style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
                />
            </div>
            <br />
            <label><b>Designated Date:</b></label>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                <input
                    type="date"
                    name="endDate"
                    value={filter.endDate}
                    onChange={handleFilterChange}
                    style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
                />
                <input
                    type="time"
                    name="endTime"
                    value={filter.endTime}
                    onChange={handleFilterChange}
                    style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
                />
            </div>
            <br />
            <div style={{ display: 'flex', gap: '10px' }}>
                <button className="download-button" onClick={applyFilter}>
                    Apply Filter
                </button>
                <button
                    className="clear-filter-button"
                    onClick={() => {
                        setFilter({
                            buildingId: '',
                            toiletId: '',
                            startTime: '',
                            endTime: '',
                            genderId: '',
                            startDate: '',
                            endDate: '',
                            floorId: '',
                        });
                        setFilteredLogs(logs); // Reset the filtered logs
                    }}
                >
                    Clear Filters
                </button>
            </div>
        </div>
    )}
</div>
                </div>
                <br></br><table className="data-table">
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
    {filteredLogs.map((log) => {
        const logDate = new Date(log.logged_at);
        return (
            <tr key={log.id}>
                <td>{log.id}</td>
                <td>{log.toilet_id}</td>
                <td>{log.building?.name || 'N/A'}</td>
                <td>{log.floor?.floor_number || 'N/A'}</td>
                <td>{log.toilet?.gender_id === 1 ? 'Male' : 'Female'}</td>
                <td>{log.usage_count}</td>
                <td>
                    {logDate.toLocaleDateString(undefined, {
                        month: 'long',
                        day: 'numeric',
                        year: 'numeric',
                    })}
                    <br />
                    {logDate.toLocaleTimeString(undefined, {
                        hour: '2-digit',
                        minute: '2-digit',
                    })}
                </td>
                <td>{log.is_reset ? 'Reset' : 'Usage'}</td>
            </tr>
        );
    })}
</tbody>

                </table>
            </div>
    );
};

export default DataPage;




