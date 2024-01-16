var els = document.getElementsByClassName("tweet");
for (var z = 0; z < els.length; z++) {
  els[z].addEventListener("click", function () {
    var c = decodeURI(this.getAttribute("data-original-content"));
    if (this.parentNode != null) {
      // console.log(c);
      this.parentNode.classList.remove("squished");
      this.parentNode.innerHTML =
        "<span style='color: black;' mark='1'>" + c + "</span>";
    }
  });
}
