// Simulate page loading for 2 seconds
setTimeout(function() {
    // Redirect to the respective page after loading
    var redirectTo = localStorage.getItem('redirectTo');
    if (redirectTo) {
        localStorage.removeItem('redirectTo'); // Clear the value from localStorage
        window.location.href = redirectTo;
    }
}, 1000); // Change the time as needed


