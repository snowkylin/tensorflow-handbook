$(document).ready(function(){
    $(".admonition-title").addClass("collapsible active");
    $(".admonition-title").attr("fold", "False");
    $(".admonition").css({"overflow": "hidden"});
    $(".admonition-title").click(function(){
        if ($(this).attr("fold") == "False"){
            height = $(this).height() + 12;
            $(this).parent().css({"height": height});
            $(this).attr("fold", "True");
            $(this).removeClass('active');
        } else {
            $(this).parent().css({"height": "inherit"});
            $(this).attr("fold", "False");
            $(this).addClass('active');
        }
    });
    $(".wy-breadcrumbs").append("<button id=\"fold_switch\" fold=\"False\">折叠全部注释（Fold all admonitions）</button>");
    $("#fold_switch").click(function(){
        if ($(this).attr("fold") == "False"){
            $(".admonition-title").attr("fold", "False");
            $(".admonition-title").click();
            $(this).attr("fold", "True");
            $(this).text("展开全部注释（Expand all admonitions）");
        } else {
            $(".admonition-title").attr("fold", "True");
            $(".admonition-title").click();
            $(this).attr("fold", "False");
            $(this).text("折叠全部注释（Fold all admonitions）");            
        }
    });
    pangu.spacingElementByClassName('wy-nav-content');
    pangu.spacingElementByClassName('wy-nav-side');
    if (window.location.pathname.includes('zh_hans/')) {
        $('.wy-menu').children().slice(14, 42).remove();   
        $('.wy-side-nav-search > a').attr('href', '/zh_hans');     
    } else if (window.location.pathname.includes('zh_hant/')) {
        $('.wy-menu').children().slice(0, 14).remove();
        $('.wy-menu').children().slice(14, 28).remove();
        $('.wy-side-nav-search > a').text('簡單粗暴 TensorFlow 2').attr('href', '/zh_hant');
    } else if (window.location.pathname.includes('en/')) {
        $('.wy-menu').children().slice(0, 28).remove();
        $('.wy-side-nav-search > a').text('A Concise Handbook of TensorFlow 2').attr('href', '/en');
    } else if (window.location.pathname.includes('zh/')) {
        window.location.href = window.location.href.replace('zh/', 'zh_hans/');
    }
    //
});