const enter_btn = document.getElementById('enter-btn');
const audio_display = document.getElementById('audio-display');
const audio_control = document.getElementById('audio-control');
const desc_textara = document.getElementById('desc-textarea');
const csrf_token = document.getElementsByName('csrfmiddlewaretoken')[0];
const spinner = document.getElementById('spinner');

const server_url = `${window.location.protocol}//${window.location.host}`;

enter_btn.onclick = async function () {
    description = desc_textara.value;
    token = csrf_token.value;
    spinner.style.display = "flex";

    await $.ajax({
        type: "POST",
        url: server_url,
        headers: {
            "X-CSRFToken": token
        },
        data: {
            "description": description,
        },
        success: function (result) {
            audio_display.src = result['audio_url']
            audio_control.load()
        },
        dataType: "json"
    });

    spinner.style.display = "none";
};
