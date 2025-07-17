// main.js — Enhanced interactivity for ChatGPT-like UI
document.addEventListener('DOMContentLoaded', () => {
    console.log('LLM WebUI loaded');
    
    // Smooth scroll for chat container
    const chatContainers = document.querySelectorAll('#chat-container');
    chatContainers.forEach(container => {
        container.addEventListener('DOMNodeInserted', () => {
            container.scrollTo({
                top: container.scrollHeight,
                behavior: 'smooth'
            });
        });
    });
});