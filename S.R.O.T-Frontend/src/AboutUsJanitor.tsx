import React from 'react';
import './css/aboutUsPage.css';
import UserProfileCard from './components/UserProfileCard/UserProfileCard';
import AboutUsPic from './gallery/aboutus.jpg'


const JanitorAboutUsPage: React.FC = () => {
    return (
<div className="main-page">
    <header>
        <nav className="hotbar">
            <div className="nav-item"><a href="/janitor">Toilet</a></div>
            <div className="nav-item"><a href="/janitorabout-us">About Us</a></div>
            <div className="nav-item"><a href="/janitordata">Data</a></div>
        </nav>
    </header>
    <div className="about-us-container">
        <header className="about-us-header">
            <h1 style={{ color: 'white' }}>About Us</h1>
        </header>
        <button className="back-button-about-us" onClick={() => window.history.back()}>
                ⬅ Back to Main Page
            </button>
        <section className="white-border">
            <div className="abtuscontain">
            <div className="combined">
                    <div className="picborder">
                        <img src={AboutUsPic} alt="amogus" className="picabtus" />
                    </div>
                            <div className='mb'>
                                <h2 className="mb-3">Why Choose This Topic?</h2>
                                <p className="lead fs-4 mb-3 mb-xl-5">We've conducted a small research...</p>
                                <div className="features">
                                    <div className="feature-item">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="currentColor" className="bi bi-check-circle-fill" viewBox="0 0 16 16">
                                            <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zm-3.97-3.03a.75.75 0 0 0-1.08.022L7.477 9.417 5.384 7.323a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-.01-1.05z" />
                                        </svg>
                                        <p className="fs-5 m-0">Uses image processing to get the best accuracy</p>
                                    </div>
                                    <div className="feature-item">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="currentColor" className="bi bi-check-circle-fill" viewBox="0 0 16 16">
                                            <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zm-3.97-3.03a.75.75 0 0 0-1.08.022L7.477 9.417 5.384 7.323a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-.01-1.05z" />
                                        </svg>
                                        <p className="fs-5 m-0">Battery efficient</p>
                                    </div>
                                    <div className="feature-item">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="currentColor" className="bi bi-check-circle-fill" viewBox="0 0 16 16">
                                            <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zm-3.97-3.03a.75.75 0 0 0-1.08.022L7.477 9.417 5.384 7.323a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-.01-1.05z" />
                                        </svg>
                                        <p className="fs-5 m-0">User & Data friendly!</p>
                                    </div>
                                </div>
                                <button type="button" className="btn bsb-btn-x200 rounded-pill" onClick={() => window.location.href='/MainPage'}>Check it out now!</button>
                                </div>
                                </div>
                                    </div>
                                    <div className="col-12 col-lg-6">
                                        <div className="row justify-content-xl-center">
                                            <div className="col-12 col-xl-10">
                                                <div className="dev-intro"> Meet Our Devs! </div>
                                                <div className="App">
                                    {<UserProfileCard />}
                                </div>
                    </div>
                </div>
            </div>
        </section>
    </div>
</div>

    );
};

export default JanitorAboutUsPage;