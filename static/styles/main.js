function addFields() {
    // Generate a dynamic number of inputs
    var number = document.getElementById("member").value;
    // Get the element where the inputs will be added to
    var container = document.getElementById("member-field");
    // Remove every children it had before
    while (container.hasChildNodes()) {
        container.removeChild(container.lastChild);
    }
    if (number > 10) {
        for (i = 0; i < 10; i++) {

            //Create div for container contain of label and input
            var formGroup = container.appendChild(document.createElement("div"));
            formGroup.className = "form-group col-md-6";

            // Append label for member name
            var label = formGroup.appendChild(document.createElement("label"));
            label.textContent = "Member " + (i + 1);

            // Create an <input> element, set its type and name attributes
            var input = document.createElement("input");
            input.type = "text";
            input.name = "memberName";
            input.className = "form-control"
            formGroup.appendChild(input);
        }
    } else {
        for (i = 0; i < number; i++) {

            //Create div for container contain of label and input
            var formGroup = container.appendChild(document.createElement("div"));
            formGroup.className = "form-group col-md-6";

            //Append label for member name
            var label = formGroup.appendChild(document.createElement("label"));
            if (i == 0) {
                label.textContent = "Ketua Rapat"
            } else {
                label.textContent = "Member " + (i);
            }
            // Create an <input> element, set its type and name attributes
            var input = document.createElement("input");
            input.type = "text";
            input.name = "memberName";
            input.setAttribute('required', '');
            input.className = "form-control"

            formGroup.appendChild(input);
        }
    }
}

function toggleScore(){
    var x = document.getElementById("score");
    if (x.style.display === "none") {
        x.style.display = "block";
    } else{
        x.style.display = "none";
    }
}


function progressBar() {
    console.log(duration);
    var bar = new ProgressBar.Line(container, {
        strokeWidth: 1,
        easing: 'easeInOut',
        duration: parseInt(duration * 185) + 30000,
        color: '#60aa3f',
        trailColor: '#eee',
        trailWidth: 4,
        svgStyle: {
            width: '100%',
            height: '100%'
        },
        text: {
            style: {
                // Text color.
                // Default: same as stroke color (options.color)
                color: '#999',
                position: 'absolute',
                right: '0',
                top: '30px',
                padding: 0,
                margin: 0,
                transform: null
            },
            autoStyleContainer: false
        },
        from: {
            color: '#FFEA82'
        },
        to: {
            color: '#ED6A5A'
        },
        step: (state, bar) => {
            bar.setText(Math.round(bar.value() * 100) + ' %');
        }
    });

    bar.animate(1.0); // Number from 0.0 to 1.0
}