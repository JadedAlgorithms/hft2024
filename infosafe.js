function getGradientColor(value) {
    const normalizedValue = value / 100;
  
    if (normalizedValue <= 0.5) {
      // Red to Yellow
      const segmentValue = normalizedValue * 2; 
      const red = 255;
      const green = Math.floor(segmentValue * 255);
      return `rgb(${red}, ${green}, 0)`; 
    } else {
      // Yellow to Green
      const segmentValue = (normalizedValue - 0.5) * 2;
      const red = Math.floor(255 - (segmentValue * 255));
      const green = 255;
      return `rgb(${red}, ${green}, 0)`;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs.length > 0) {
        const activeTab = tabs[0];
        const activeTabUrl = activeTab.url;
  
        chrome.runtime.sendMessage(
          { action: "fetchData", payload: { query: activeTabUrl } },
          (response) => {
            if (chrome.runtime.lastError) {
              document.body.style.backgroundColor = 'whitesmoke';
              document.getElementById("data").textContent =
                "Error: " + chrome.runtime.lastError.message;
              return;
            }
  
            if (response && response.data) {
              document.getElementById("data").textContent = `Similarity score: ${response.data}%`
              document.body.style.backgroundColor = getGradientColor(response.data);
            } else {
              document.getElementById("data").textContent = "No data received.";
              document.body.style.backgroundColor = 'whitesmoke';
            }
          }
        );
      } else {
        document.getElementById("data").textContent = "No active tab found.";
        document.body.style.backgroundColor = 'whitesmoke';
      }
    });
});