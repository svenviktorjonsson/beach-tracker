// Global variables
let video, clock, startButton, stopButton, playbackVideo, videoTime;
let startTime = null;
let clockInterval = null;
let videoTimeInterval = null;
let mediaRecorder = null;
let recordedChunks = [];
let videoDatabase = [];
let videoDurations = [];


// Menu functions - make them global
function showMainMenu() {
    document.getElementById('mainMenu').classList.remove('hidden');
    document.getElementById('videoContainer').classList.add('hidden');
    document.getElementById('analysisContainer').classList.add('hidden');
    document.getElementById('statisticsContainer').classList.add('hidden');
}

function showAnalysis() {
    document.getElementById('mainMenu').classList.add('hidden');
    document.getElementById('analysisContainer').classList.remove('hidden');
}

function showRecord() {
    document.getElementById('mainMenu').classList.add('hidden');
    document.getElementById('videoContainer').classList.remove('hidden');
}

function showStatistics() {
    document.getElementById('mainMenu').classList.add('hidden');
    document.getElementById('statisticsContainer').classList.remove('hidden');
}

function playPause() {
    const playbackVideo = document.getElementById('playbackVideo');
    if (playbackVideo.paused) {
        playbackVideo.play();
    } else {
        playbackVideo.pause();
    }
}

function changeSpeed() {
    const speedSelector = document.getElementById('speedSelector');
    const playbackVideo = document.getElementById('playbackVideo');
    playbackVideo.playbackRate = parseFloat(speedSelector.value);
}

// Make functions available globally
window.showMainMenu = showMainMenu;
window.showAnalysis = showAnalysis;
window.showRecord = showRecord;
window.showStatistics = showStatistics;
window.playPause = playPause;
window.changeSpeed = changeSpeed;