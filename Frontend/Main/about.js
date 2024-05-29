function redirectTo(page) {
    localStorage.setItem('redirectTo', page);
}


document.addEventListener('DOMContentLoaded', function() {
    const slider = document.querySelector('.slider');
    const translators = slider.querySelectorAll('.translator');
    const prevBtn = document.querySelector('.nav-btn.prev');
    const nextBtn = document.querySelector('.nav-btn.next');

    let slideIndex = 0;

    prevBtn.addEventListener('click', function() {
        if (slideIndex > 0) {
            slideIndex--;
            slide();
        }
    });

    nextBtn.addEventListener('click', function() {
        if (slideIndex < translators.length - 1) {
            slideIndex++;
            slide();
        }
    });

    function slide() {
        translators.forEach((translator, index) => {
            if (index === slideIndex) {
                translator.style.display = 'block';
            } else {
                translator.style.display = 'none';
            }
        });
    }

    slide(); 
});
