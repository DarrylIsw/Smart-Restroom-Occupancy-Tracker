import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Login from './Login';
import MainPage from './MainPage';
import DataPage from './components/DataPage';
import AboutUsPage from './AboutUs';
import JanitorMainPage from './JanitorMainPage';
import JanitorDataPage from './components/JanitorData';
import JanitorAboutUsPage from './AboutUsJanitor';


const App: React.FC = () => {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<Login />} />
                <Route path="/admin0x0x0x0x" Component={MainPage} />
                <Route path="/data" element={<DataPage />} />
                <Route path="/about-us" element={<AboutUsPage />} />
                <Route path="/janitor" element={<JanitorMainPage />} />
                <Route path="/janitordata" element={<JanitorDataPage />} />
                <Route path="/janitorabout-us" element={<JanitorAboutUsPage />} />
            </Routes>
        </Router>
    );
};

export default App;
