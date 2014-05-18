$(document).ready(function() {
  $("#navlist li a").each(function(index) {
    if ($(this).attr("href") == window.location.pathname)
    $(this).parent().addClass("active");
  });
});
