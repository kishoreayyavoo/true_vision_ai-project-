chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "startLiveDetection") {
    console.log("Starting live detection...");

    let extractionInterval; 

    const video = document.querySelector('video');

    if (!video) {
      console.error("No video element found on the page.");
      sendResponse({ status: "No video found!" });
      return;
    }
    video.addEventListener('pause', () => {
      console.log("Video paused. Stopping frame extraction.");
      clearInterval(extractionInterval); 
    });
    extractionInterval = setInterval(() => {
      console.log("Extracting frames...");
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob);
          const downloadLink = document.createElement("a");
          downloadLink.href = url;
          downloadLink.download = `frame_${Date.now()}.png`; 
          downloadLink.click(); 
          URL.revokeObjectURL(url); 
        } else {
          console.error("Failed to convert canvas to blob.");
        }
      }, "image/png"); 
    }, 5000); 

    sendResponse({ status: "Live detection started!" });
  }
});
