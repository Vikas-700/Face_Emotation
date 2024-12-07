document.addEventListener('DOMContentLoaded', () => {
    console.log("Emotion detection app is loaded and ready!");

  

    // Display message on closing video feed
    const videoElement = document.querySelector('img');
    if (videoElement) {
        videoElement.addEventListener('error', () => {
            alert("Error: Unable to load the video feed. Please check your webcam or server.");
        });
    }
});
