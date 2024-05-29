// Function to parse query parameters from URL
function getQueryParams() {
    const queryParams = {};
    const queryString = window.location.search.substring(1);
    const pairs = queryString.split("&");
    for (let i = 0; i < pairs.length; i++) {
        const pair = pairs[i].split("=");
        queryParams[decodeURIComponent(pair[0])] = decodeURIComponent(pair[1]);
    }
    return queryParams;
}

// Function to create and display "Form Submitted" 200px up from the bottom right corner
function displayMessage() {
    const queryParams = getQueryParams();
    const message = queryParams["message"];
    if (message === "FormSubmitted") {
        const messageDiv = document.createElement("div");
        messageDiv.textContent = "Form Submitted";
        messageDiv.style.position = "fixed";
        messageDiv.style.bottom = "220px"; // 200px + 20px padding
        messageDiv.style.right = "20px";
        messageDiv.style.backgroundColor = "rgba(0, 0, 0, 0.8)";
        messageDiv.style.color = "#ffffff";
        messageDiv.style.padding = "10px";
        messageDiv.style.borderRadius = "5px";
        document.body.appendChild(messageDiv);
    }
}
function redirectToContactPage() {
    window.location.href = "contact.html?message=FormSubmitted";
}
// Call the function when the page loads
window.onload = function() {
    displayMessage();
};
