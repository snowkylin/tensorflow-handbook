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
    $(".wy-breadcrumbs").append("<button id=\"translateLink\" onclick=\"javascript:translatePage();\">切換到繁體</button>");
    var defaultEncoding = 2; //网站编写字体是否繁体，1-繁体，2-简体
    var translateDelay = 0; //延迟时间,若不在前, 要设定延迟翻译时间, 如100表示100ms,默认为0
    var cookieDomain = "https://tf.wiki/"; //Cookie地址, 一定要设定, 通常为你的网址
    var msgToTraditionalChinese = "切換到繁體"; //此处可以更改为你想要显示的文字
    var msgToSimplifiedChinese = "切换到简体"; //同上，但两处均不建议更改
    var translateButtonId = "translateLink"; //默认互换id
    translateInitilization();
});