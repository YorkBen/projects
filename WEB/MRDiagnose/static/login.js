$(function () {
  var loadUsrInfo = function(userid) {
    if (userid == "test") {
      // 请求信息科，抓取姓名
      username = '测试用户'

      // 姓名字段赋值
      $('#username').val(username)

      // 请求信息科，抓取科室数据
      let depart_list = [
        ['消化内科门诊','消化内科门诊'],
        ['消化内科住院','消化内科住院']
      ]

      // 修改下拉菜单
      $('#depart').empty()
      for(let i = 0; i < depart_list.length; i++) {
        depart = depart_list[i]
        console.log(depart)
        $("#depart").append("<option value='" + depart[0] + "'>" + depart[1] + "</option>");
      }
    }
  }
  $(document).ready(function(){
    if ($('#userid').val() != '') {
      loadUsrInfo($('#userid').val());
    }
  });

  $('#userid').change(function(){
    loadUsrInfo($('#userid').val());
  });
});
