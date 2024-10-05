chrome.runtime.onInstalled.addListener(() => {
  console.log("Background script running...");
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "downloadFrame") {
    const url = message.dataUrl;

    chrome.downloads.download(
      {
        url: url,
        filename: `frame-${Date.now()}.png`,
        saveAs: false,
      },
      function (downloadId) {
        console.log(`Frame saved with ID: ${downloadId}`);
      }
    );
  }
});
