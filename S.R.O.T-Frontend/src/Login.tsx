import React, { useState } from 'react';
import axios, { AxiosError } from 'axios';
import { useNavigate } from 'react-router-dom';
import './css/Login.css';

interface LoginResponse {
    role: string; // Expected role from the backend
}

interface ErrorResponse {
    message: string; // Define the expected shape of the error response
}

const Login: React.FC = () => {
    const [username, setUsername] = useState<string>('');
    const [password, setPassword] = useState<string>('');
    const [error, setError] = useState<string>('');
    const navigate = useNavigate();

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(''); // Clear previous error messages
    
        try {
            const response = await axios.post('http://localhost:8000/api/login', { username, password });
    
            // Get user role from the response
            const { role } = response.data;
    
            // Redirect based on role
            if (role === 'admin') {
                navigate('/admin0x0x0x0x'); // Admin main page
            } else if (role === 'janitor') {
                navigate('/janitor'); // Janitor main page
            }
        } catch (error) {
            // Handle API errors
            const axiosError = error as AxiosError<ErrorResponse>;
            if (axiosError.response && axiosError.response.data) {
                setError(axiosError.response.data.message || 'Login failed. Please check your credentials.');
            } else {
                setError('An unexpected error occurred.');
            }
        }
    };
    

    return (
        <div className="container">
            <div className="form-container sign-in">
                <form onSubmit={handleLogin}>
                    <h2>Sign in</h2>
                    {error && <p style={{ color: 'red' }}>{error}</p>}
                    <input
                        type="text"
                        placeholder="Username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        required
                    />
                    <input
                        type="password"
                        placeholder="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                    <button type="submit">Login</button>
                </form>
            </div>

            <div className="toggle-container">
                <div className="toggle">
                    <div className="toggle-panel toggle-right">
                        <h2>Hello, UMN Staff!</h2>
                        <p>Please Enter Your Login Credentials To Access The Main Page</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Login;





