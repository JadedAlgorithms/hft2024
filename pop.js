document.getElementById("checkBtn").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      let url = tabs[0].url;
      alert("Checking trust for: " + url);
      // Add your trust-checking logic here
    });
  });
  