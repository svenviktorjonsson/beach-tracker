console.log("--- script.js: Execution started ---");

// --- Global Variables ---
let videoStream, mediaRecorder, recordedChunks = [];
let clockInterval, startTime;

// Updated placeholder data - FPS is assumed/known for these
let allRecordedVideos = [
    { id: 'vid1', name: "Demo Game 1 Highlights", timestamp: "2025-05-28 14:30", src: "https://www.w3schools.com/html/mov_bbb.mp4", duration: 10.2, width: 320, height: 240, fps: 25 },
    { id: 'vid2', name: "Practice Drills - Serving", timestamp: "2025-05-29 09:15", src: "https://www.w3schools.com/tags/movie.mp4", duration: 4.8, width: 640, height: 360, fps: 30 },
    { id: 'vid3', name: "Fun Match Play", timestamp: "2025-05-30 18:00", src: "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.webm", duration: 17.23, width: 1920, height: 1080, fps: 24 }
];

// --- DOM Element References ---
// (These remain the same as your last complete script.js)
const mainMenuEl = document.getElementById('mainMenu');
const analysisContainerEl = document.getElementById('analysisContainer');
const videoContainerEl = document.getElementById('videoContainer');
const statisticsContainerEl = document.getElementById('statisticsContainer');
const videoPlaceholderEl = document.getElementById('videoPlaceholder');
const playbackVideoEl = document.getElementById('playbackVideo');
const videoTimeEl = document.getElementById('videoTime');
const customSeekBarContainerEl = document.getElementById('customSeekBarContainer');
const customSeekBarEl = document.getElementById('customSeekBar');
const speedSelectorEl = document.getElementById('speedSelector');
const videoFileInputEl = document.getElementById('videoFileInput');
const videoSearchInputEl = document.getElementById('videoSearchInput');
const videoListContainerEl = document.getElementById('videoListContainer');
const liveVideoEl = document.getElementById('video');
const clockDisplayEl = document.getElementById('clock');
const startButtonEl = document.getElementById('startButton');
const stopButtonEl = document.getElementById('stopButton');
const statsSearchInputEl = document.getElementById('statsSearchInput');
const statsVideoCountEl = document.getElementById('statsVideoCount');
const statsTableBodyEl = document.getElementById('statsTable')?.querySelector('tbody');

let isUserSeeking = false;

// --- Navigation / View Management ---
// (setActiveView, showMainMenu, showRecord remain the same)
const views = ['mainMenu', 'analysisContainer', 'videoContainer', 'statisticsContainer'];
function setActiveView(activeViewId) {
    console.log("Setting active view to:", activeViewId);
    views.forEach(viewId => {
        const element = document.getElementById(viewId);
        if (element) {
            element.classList.toggle('hidden', viewId !== activeViewId);
        } else {
            console.warn(`Element with ID '${viewId}' not found during view switch.`);
        }
    });
}

function showMainMenu() {
    setActiveView('mainMenu');
    if (playbackVideoEl && !playbackVideoEl.paused) {
        playbackVideoEl.pause();
    }
}

function showRecord() {
    setActiveView('videoContainer');
}


// --- Modified showAnalysis and showStatistics ---
function showAnalysis() {
    setActiveView('analysisContainer');
    if (videoSearchInputEl) videoSearchInputEl.value = '';
    renderVideoList(allRecordedVideos); // Display videos (now potentially with updated metadata)

    if (playbackVideoEl) {
        playbackVideoEl.style.display = 'none';
        playbackVideoEl.src = '';
    }
    if (videoPlaceholderEl) videoPlaceholderEl.style.display = 'flex';
    if (videoTimeEl) videoTimeEl.style.display = 'none';
    if (customSeekBarContainerEl) customSeekBarContainerEl.style.display = 'none';
    if (customSeekBarEl) customSeekBarEl.value = 0;
    console.log("Analysis view shown, video player UI reset.");
}

function showStatistics() {
    setActiveView('statisticsContainer');
    if (statsSearchInputEl) statsSearchInputEl.value = '';
    filterAndCalculateStats(); // Initial calculation
}

// --- Video Metadata Extraction ---
function extractVideoMetadata(videoSrc) {
    return new Promise((resolve, reject) => {
        const tempVideo = document.createElement('video');
        tempVideo.preload = 'metadata'; // Only need metadata
        tempVideo.muted = true;
        tempVideo.playsInline = true;

        tempVideo.onloadedmetadata = () => {
            console.log(`Metadata extracted: D:${tempVideo.duration}, W:${tempVideo.videoWidth}, H:${tempVideo.videoHeight}`);
            resolve({
                duration: tempVideo.duration,
                width: tempVideo.videoWidth,
                height: tempVideo.videoHeight,
                // FPS is not reliably available here. It will be a placeholder.
            });
            // Cleanup
            tempVideo.src = ''; // Release resources
            tempVideo.removeAttribute('src');
            tempVideo.load();
        };

        tempVideo.onerror = (e) => {
            let errorMsg = "Unknown error loading video metadata.";
            if (tempVideo.error) {
                switch (tempVideo.error.code) {
                    case tempVideo.error.MEDIA_ERR_ABORTED: errorMsg = 'Video loading aborted.'; break;
                    case tempVideo.error.MEDIA_ERR_NETWORK: errorMsg = 'Network error during video loading.'; break;
                    case tempVideo.error.MEDIA_ERR_DECODE: errorMsg = 'Error decoding video.'; break;
                    case tempVideo.error.MEDIA_ERR_SRC_NOT_SUPPORTED: errorMsg = 'Video source not supported.'; break;
                    default: errorMsg = 'An unknown error occurred.';
                }
            }
            console.error("Error in tempVideo:", errorMsg, e);
            reject(new Error(errorMsg));
            // Cleanup
            tempVideo.src = '';
            tempVideo.removeAttribute('src');
            tempVideo.load();
        };

        tempVideo.src = videoSrc;
        // Note: No need to call tempVideo.load() explicitly after setting src, it loads automatically.
    });
}


// --- Analysis View Functions ---
// (renderVideoList, loadAnalysisVideo, filterAndDisplayVideos, formatTime, playPause, changeSpeed)
// `loadAnalysisVideo` and `renderVideoList` will use the `allRecordedVideos` which now can be updated with extracted metadata.
// The existing versions of these functions are generally fine.
function renderVideoList(videosToDisplay) {
    if (!videoListContainerEl) { console.warn("videoListContainerEl not found for rendering."); return; }
    videoListContainerEl.innerHTML = '';
    if (!videosToDisplay || videosToDisplay.length === 0) {
        videoListContainerEl.innerHTML = '<p style="text-align:center; color:#888; padding:10px;">No videos found.</p>';
        return;
    }
    videosToDisplay.forEach(video => {
        const videoItem = document.createElement('div');
        videoItem.className = 'video-list-item';
        const safeName = video.name.replace(/'/g, "\\'").replace(/"/g, "&quot;");
        videoItem.setAttribute('onclick', `loadAnalysisVideo('${video.src}', '${safeName}')`);
        const nameSpan = document.createElement('span'); nameSpan.className = 'video-name'; nameSpan.textContent = video.name; nameSpan.title = video.name;
        const timestampSpan = document.createElement('span'); timestampSpan.className = 'video-timestamp'; timestampSpan.textContent = typeof video.timestamp === 'string' ? video.timestamp : video.timestamp.toLocaleString();
        videoItem.appendChild(nameSpan); videoItem.appendChild(timestampSpan);
        videoListContainerEl.appendChild(videoItem);
    });
}
function loadAnalysisVideo(videoSrc, videoName) {
    if (!playbackVideoEl || !videoPlaceholderEl || !videoTimeEl || !customSeekBarContainerEl || !customSeekBarEl) { console.error("Required media elements not found."); return; }
    console.log(`Attempting to load video: ${videoName} (Source: ${videoSrc})`);
    videoPlaceholderEl.style.display = 'none'; playbackVideoEl.style.display = 'block'; videoTimeEl.style.display = 'none'; customSeekBarContainerEl.style.display = 'none';
    playbackVideoEl.src = videoSrc; playbackVideoEl.load();
    playbackVideoEl.onloadedmetadata = function() {
        console.log(`Metadata loaded: ${videoName}. Duration: ${this.duration}`); this.currentTime = 0; this.pause();
        videoTimeEl.textContent = formatTime(0); videoTimeEl.style.display = 'block';
        customSeekBarEl.max = this.duration; customSeekBarEl.value = 0; customSeekBarContainerEl.style.display = 'block';
    };
    playbackVideoEl.oncanplay = function() { console.log(`Video can play: ${videoName}`); };
    playbackVideoEl.onplay = function() { console.log("Video playback started."); };
    playbackVideoEl.onpause = function() { console.log("Video playback paused."); };
    playbackVideoEl.onerror = function(e) {
        console.error("Error loading video in player:", videoSrc, e);
        playbackVideoEl.style.display = 'none'; videoTimeEl.style.display = 'none'; customSeekBarContainerEl.style.display = 'none';
        if (videoPlaceholderEl) { videoPlaceholderEl.innerHTML = '<p>Error playing this video.</p>'; videoPlaceholderEl.style.display = 'flex';}
    };
}
function filterAndDisplayVideos() { /* For Analysis View */
    if (!videoSearchInputEl || !allRecordedVideos) return;
    const searchTerm = videoSearchInputEl.value.toLowerCase();
    const filteredVideos = allRecordedVideos.filter(video =>
        video.name.toLowerCase().includes(searchTerm) ||
        (typeof video.timestamp === 'string' ? video.timestamp : video.timestamp.toLocaleString()).toLowerCase().includes(searchTerm)
    );
    renderVideoList(filteredVideos);
}
function formatTime(timeInSeconds) {
    const minutes = Math.floor(timeInSeconds / 60); const seconds = Math.floor(timeInSeconds % 60);
    const milliseconds = Math.floor((timeInSeconds % 1) * 10);
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${milliseconds}`;
}
function playPause() {
    if (!playbackVideoEl || !playbackVideoEl.src || playbackVideoEl.src === window.location.href) { console.warn("Play/Pause: No video loaded."); return; }
    if (playbackVideoEl.paused || playbackVideoEl.ended) { playbackVideoEl.play().catch(e => console.error("Play error:", e)); }
    else { playbackVideoEl.pause(); }
}
function changeSpeed() { if (playbackVideoEl && speedSelectorEl) { playbackVideoEl.playbackRate = parseFloat(speedSelectorEl.value); } }


// --- Statistics View Functions ---
// (updateStatsTableRow, calculateAndDisplayStatistics, filterAndCalculateStats remain the same from previous version)
// Ensure they are present and correct as provided in the last complete script.
function updateStatsTableRow(propertyIdPrefix, min, max, avg) {
    const minEl = document.getElementById(`${propertyIdPrefix}Min`);
    const maxEl = document.getElementById(`${propertyIdPrefix}Max`);
    const avgEl = document.getElementById(`${propertyIdPrefix}Avg`);
    if(minEl) minEl.textContent = min;
    if(maxEl) maxEl.textContent = max;
    if(avgEl) avgEl.textContent = avg;
}
function calculateAndDisplayStatistics(videos) {
    if (!statsTableBodyEl) { console.warn("Statistics table body not found."); return; }
    const count = videos.length;
    if (statsVideoCountEl) statsVideoCountEl.textContent = `Videos included: ${count}`;

    if (count === 0) {
        updateStatsTableRow('statDuration', '-', '-', '-'); updateStatsTableRow('statWidth', '-', '-', '-');
        updateStatsTableRow('statHeight', '-', '-', '-'); updateStatsTableRow('statPixels', '-', '-', '-');
        updateStatsTableRow('statFps', '-', '-', '-');
        return;
    }
    const getStat = (propExtractor, unitConverter = val => val) => {
        let min = Infinity, max = -Infinity, sum = 0, validCount = 0;
        videos.forEach(video => {
            const val = propExtractor(video);
            if (typeof val === 'number' && !isNaN(val)) {
                min = Math.min(min, val); max = Math.max(max, val); sum += val; validCount++;
            }
        });
        if (validCount === 0) return { min: '-', max: '-', avg: '-' };
        const formatNumber = num => num.toLocaleString(undefined, {minimumFractionDigits: (num % 1 === 0 ? 0 : 1), maximumFractionDigits: 2});
        return { min: formatNumber(unitConverter(min)), max: formatNumber(unitConverter(max)), avg: formatNumber(unitConverter(sum / validCount)) };
    };
    const durationStats = getStat(v => v.duration);
    const widthStats = getStat(v => v.width);
    const heightStats = getStat(v => v.height);
    const pixelsStats = getStat(v => (v.width && v.height) ? (v.width * v.height) : undefined, p => p / 1000000); // MP
    const fpsStats = getStat(v => v.fps);

    updateStatsTableRow('statDuration', durationStats.min, durationStats.max, durationStats.avg);
    updateStatsTableRow('statWidth', widthStats.min, widthStats.max, widthStats.avg);
    updateStatsTableRow('statHeight', heightStats.min, heightStats.max, heightStats.avg);
    updateStatsTableRow('statPixels', pixelsStats.min, pixelsStats.max, pixelsStats.avg);
    updateStatsTableRow('statFps', fpsStats.min, fpsStats.max, fpsStats.avg);
}
function filterAndCalculateStats() {
    if (!statsSearchInputEl || !allRecordedVideos) { calculateAndDisplayStatistics([]); return; }
    const searchTerm = statsSearchInputEl.value.toLowerCase();
    const filteredVideos = allRecordedVideos.filter(video => {
        if (!video || typeof video.name !== 'string' || typeof video.timestamp === 'string') return false;
        return video.name.toLowerCase().includes(searchTerm) || video.timestamp.toLowerCase().includes(searchTerm);
    });
    calculateAndDisplayStatistics(filteredVideos);
}


// --- Event Listeners ---
// Analysis View related event listeners (playbackVideoEl, customSeekBarEl, videoSearchInputEl)
// These remain the same.
if (playbackVideoEl) {
    playbackVideoEl.addEventListener('timeupdate', function() {
        if (videoTimeEl && videoTimeEl.style.display === 'block') { videoTimeEl.textContent = formatTime(this.currentTime); }
        if (customSeekBarEl && !isUserSeeking) { customSeekBarEl.value = this.currentTime; }
    });
}
if (customSeekBarEl) {
    customSeekBarEl.addEventListener('mousedown', () => { isUserSeeking = true; });
    customSeekBarEl.addEventListener('touchstart', () => { isUserSeeking = true; }, { passive: true });
    customSeekBarEl.addEventListener('input', function() {
        if (playbackVideoEl && playbackVideoEl.readyState >= playbackVideoEl.HAVE_METADATA) {
            playbackVideoEl.currentTime = parseFloat(this.value);
            if (videoTimeEl && videoTimeEl.style.display === 'block') { videoTimeEl.textContent = formatTime(playbackVideoEl.currentTime); }
        }
    });
    customSeekBarEl.addEventListener('mouseup', () => { isUserSeeking = false; });
    customSeekBarEl.addEventListener('touchend', () => { isUserSeeking = false; });
}
if (videoSearchInputEl) { videoSearchInputEl.addEventListener('input', filterAndDisplayVideos); }

// Statistics view search listener
if (statsSearchInputEl) { statsSearchInputEl.addEventListener('input', filterAndCalculateStats); }

// --- Modified File Input Listener to Extract Metadata ---
if (videoFileInputEl) {
    videoFileInputEl.addEventListener('change', async function(event) { // Made async
        const file = event.target.files[0];
        if (file) {
            const fileURL = URL.createObjectURL(file);
            let newVideoEntry; // To hold the new video object

            try {
                console.log("Extracting metadata for:", file.name);
                const metadata = await extractVideoMetadata(fileURL);
                console.log("Extracted metadata:", metadata);

                newVideoEntry = {
                    id: 'file-' + Date.now() + '-' + Math.random().toString(36).substring(2, 11),
                    name: file.name,
                    timestamp: new Date().toLocaleString(), // Use current date/time
                    src: fileURL, // This is a Blob URL
                    duration: metadata.duration,
                    width: metadata.width,
                    height: metadata.height,
                    fps: 25 // Placeholder FPS - actual extraction is complex.
                           // You might allow users to edit this or set a common default.
                };

                allRecordedVideos.push(newVideoEntry);
                renderVideoList(allRecordedVideos); // Update analysis list
                
                // If currently in statistics view, update stats
                if (statisticsContainerEl && !statisticsContainerEl.classList.contains('hidden')) {
                    filterAndCalculateStats();
                }

                loadAnalysisVideo(newVideoEntry.src, newVideoEntry.name); // Load for playback in analysis view

            } catch (error) {
                console.error("Failed to extract metadata or add video:", error);
                alert("Could not process video: " + error.message);
                // If metadata extraction failed, the blob URL might still be active
                // and not yet passed to loadAnalysisVideo.
                // It's generally good to revoke blob URLs when no longer needed,
                // but loadAnalysisVideo will use it. If that fails, its own error handler
                // might be a place, or if this whole block fails before loadAnalysisVideo.
                // For simplicity, not revoking here as loadAnalysisVideo is next.
            }
            event.target.value = null; // Reset file input
        }
    });
}


// --- Recording Functionality (Placeholders) ---
if (startButtonEl) {
    startButtonEl.onclick = async () => {
        alert("Recording functionality not fully implemented yet.");
    };
}
if (stopButtonEl) {
    stopButtonEl.onclick = () => {
        alert("Recording stop functionality not fully implemented yet.");
    };
}

// --- Make functions globally available for HTML onclick attributes ---
window.showMainMenu = showMainMenu;
window.showAnalysis = showAnalysis;
window.showRecord = showRecord;
window.showStatistics = showStatistics;
window.playPause = playPause;
window.changeSpeed = changeSpeed;
window.loadAnalysisVideo = loadAnalysisVideo;

// --- Initial Setup ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed. Initializing application.");
    setActiveView('mainMenu');
});

console.log("--- script.js: Initial execution finished ---");