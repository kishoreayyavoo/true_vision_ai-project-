document.getElementById("liveDetection").addEventListener("click", () => {
  console.log("Button clicked, attempting to send message to content script");

  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    if (tabs.length === 0) {
      console.error("No active tab found.");
      return;
    }
    chrome.scripting.executeScript(
      {
        target: { tabId: tabs[0].id },
        files: ["content.js"],
      },
      () => {
        chrome.tabs.sendMessage(
          tabs[0].id,
          { action: "startLiveDetection" },
          function (response) {
            if (chrome.runtime.lastError) {
              console.error(
                "Error sending message: ",
                chrome.runtime.lastError.message
              );
            } else {
              console.log("Message sent successfully: ", response);
            }
          }
        );
      }
    );
  });
});
