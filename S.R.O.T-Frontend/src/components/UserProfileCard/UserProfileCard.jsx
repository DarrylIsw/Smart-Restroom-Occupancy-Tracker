import React from 'react';
import './UserProfileCard.css';
import profile_icon from '../Assets/Dev.png'
import profile_icon2 from '../Assets/Dev2.png'
import profile_icon3 from '../Assets/Dev3.png'
import profile_icon4 from '../Assets/Dev4.png'

export const UserProfileCard = () => {
return (
    <>
        <div className="profiles-container">
            <div className="zidan">
            <div className='zidgradient'></div>
            <div className='zidupc'>
                <div className='profile-down'> 
                    <img src={profile_icon} alt="Developer" />
                    <div className="profile-title">Zidane</div>
                    <div className="profile-description">
                        That's Crazy!
                    </div>
                    <div className="profile-button">
                        <a href="https://www.linkedin.com/in/danish-zidane/">LinkedIn</a>
                    </div>
                </div>
            </div>
            </div>
            <div className='ryan'>
                <div className='yangradient2'></div>
                <div className='yanupc'>
                    <div className='profile-down'> 
                        <img src={profile_icon2} alt="Developer" />
                        <div className="profile-title">Ryan</div>
                        <div className="profile-description">
                            Blm buat bgt aye
                        </div>
                        <div className="profile-button">
                            <a href="https://www.linkedin.com/in/jonathan-rizkianto-a0b22028a/">LinkedIn</a>
                        </div>
                    </div>
            </div>
            </div>
            <div className='Matt'>
            <div className='gradient'></div>
            <div className='upc'>
                <div className='profile-down'> 
                    <img src={profile_icon3} alt="Developer" />
                    <div className="profile-title">Matthew</div>
                    <div className="profile-description">
                    Bntar gw buat dl
                    </div>
                    <div className="profile-button">
                        <a href="https://www.linkedin.com/in/matthew-kusnandar-2a64a033a/ ">LinkedIn</a>
                    </div>
                </div>
            </div>
            </div>
            <div className='Dar'>
            <div className='gradient2'></div>
            <div className='upc'>
                <div className='profile-down'> 
                    <img src={profile_icon4} alt="Developer" />
                    <div className="profile-title">Darryl</div> {/*ganteng*/} 
                    <div className="profile-description">
                    "jangan masuk tekkom"
                    </div>
                    <div className="profile-button">
                        <a href="https://www.linkedin.com/in/darryl-pratama-iswahyudhi-71aba6339/">LinkedIn</a>
                    </div>
                </div>
            </div> 
            </div>
        </div>
    </>
);
};

export default UserProfileCard;
