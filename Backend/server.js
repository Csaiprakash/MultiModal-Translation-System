const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const path = require('path');

const app = express();

// Connect to MongoDB
mongoose.connect('mongodb+srv://csaiprakash77:saiprakash123@cluster0.zkyo2dd.mongodb.net/Translator', { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => {
        console.log("Connected to MongoDB");
    })
    .catch((error) => {
        console.error("MongoDB connection error:", error);
    });

// Define submission schema and model
const submissionSchema = new mongoose.Schema({
    name: String,
    email: String,
    query: String
});
const Submission = mongoose.model("Submission", submissionSchema);

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files from the 'Frontend/Main' folder
app.use(express.static(path.join(__dirname, '..', 'Frontend', 'Main')));

// Serve Text Translator.html for the specified URL


// Handle form submission
app.post("/submit", function(req, res) {
    let newSubmission = new Submission({
        name: req.body.name,
        email: req.body.email,
        query: req.body.query
    });
    newSubmission.save()
        .then(() => {
            console.log("Form submitted successfully");
            // Redirect to contact.html after successful submission
            res.redirect("/contact.html");
        })
        .catch((error) => {
            console.error("Error submitting form:", error);
            res.status(500).send("Error submitting form");
        });
});




// Serve Translator.html for the root URL
app.get("/", function(req, res) {
    res.sendFile(path.join(__dirname, '..', 'Frontend', 'Main', 'Translator.html'));
});



app.get("/about", function(req, res) {
    res.sendFile(path.join(__dirname, '..', 'Frontend', 'Main', 'About.html'));
});

// Serve Services.html
app.get("/services", function(req, res) {
    res.sendFile(path.join(__dirname, '..', 'Frontend', 'Main', 'Services.html'));
});

// Serve Contact.html
app.get("/contact", function(req, res) {
    res.sendFile(path.join(__dirname, '..', 'Frontend', 'Main', 'Contact.html'));
});



app.get("/text-translator", function(req, res) {
    res.sendFile(path.join(__dirname, '..', 'Frontend', 'Sign & Language','Text Translator.html'));
});

// Serve Document Translator.html when the "Document Text Extraction and Translation" button is clicked
app.get("/document-translator", function(req, res) {
    res.sendFile(path.join(__dirname, '..', 'Frontend', 'Sign & Language','Document Translator.html'));
});



// Serve index.html from the templates folder when the "/sign-translator" endpoint is accessed
app.get("/sign-translator", function(req, res) {
    res.sendFile(path.join(__dirname, '..', 'SignLanguage', 'templates', 'index.html'));
});




// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, function() {
    console.log(`Server is running on port ${PORT}`);
});










