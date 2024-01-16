chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
  if (request.message == "listeners") {
    console.log("background now");

    //add event handler for button click
    await chrome.tabs.executeScript(null, { file: "injectedScript.js" });
    sendResponse({ status: true });
  }
});
