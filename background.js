chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "fetchData") {
    const apiUrl = "https://b265-34-148-89-27.ngrok-free.app/ask";
    const formData = new URLSearchParams();
    formData.append("user_url", message.payload.query); // Add the active tab's URL

    fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: formData.toString(),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        return response.json(); // Parse JSON response
      })
      .then((data) => {
        if (data.similarity_score !== undefined) {
          sendResponse({ data: data.similarity_score });
        } else {
          sendResponse({ data: "No similarity score returned." });
        }
      })
      .catch((error) => {
        console.error("Error fetching API:", error);
        sendResponse({ data: "Error fetching data." });
      });

    return true; // Keeps the message channel open for async response
  }
});
