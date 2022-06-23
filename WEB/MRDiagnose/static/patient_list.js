var PatientTable = {
  createNew: function(table_id, pageSize) {
    var pTable = {}
    pTable.id = table_id
    pTable.pageSize = pageSize

    // 加载数据，第一次进入页面时请求
    pTable.loadData = function() {
      $.ajax({
    		url:"/patient_list_data",
        headers: {
          Accept: "application/json; charset=utf-8"
        },
    		data:{action:"jQueryAjax"},
    		type:"GET",
    		success:function (data) {
          if(data.status_code == 200) {
            pTable.data = data.text;
            pTable.curData = pTable.data;
            pTable.totalPages = Math.ceil(pTable.curData.length / pTable.pageSize);
            pTable.gotopage(0);
          }
          else {
            alert(data['reason']);
          }
    		},
    		//返回数据的格式
    		dataType : "json"
    	});
    }

    // 抓取显示数据
    pTable.search = function(patientName) {
      patientName = patientName.trim();
      pTable.curData = []
      for (let i=0; i < pTable.data.length; i++) {
        if (pTable.data[i][2].indexOf(patientName) >= 0) {
          pTable.curData.push(pTable.data[i])
        }
      }
      pTable.totalPages = Math.ceil(pTable.curData.length / pTable.pageSize);
    }

    // 显示数据
    pTable.showPage = function(pageNo) {
      if (pageNo >= pTable.totalPages) {
        alert('page no illegal!')
        return false;
      }

      let table_html = ''
      for(let i=pageNo*pTable.pageSize; i<Math.min(pTable.curData.length, (pageNo+1)*pTable.pageSize); i++){
        if (pTable.curData[i].length > 0) {
          table_html += '<tr>'
          table_html += '<th scope="row">' + pTable.curData[i][0] + '</th>'
          for (let j=1; j<pTable.curData[i].length-1; j++){
            table_html += '<td>' + pTable.curData[i][j] + '</td>'
          }
          table_html += '<td><a href="/patient/' + pTable.curData[i][1] + '" target="_blank" />详情</td>'
          table_html += '</tr>'
        }
      }
      $('#' + pTable.id + ' tbody').html(table_html)
    }

    // 显示分页按钮
    pTable.showPagination = function(pageNo) {
      nav_html = ''
      for(let i=0; i<pTable.totalPages; i++) {
        if (pageNo == i) {
          nav_html += '<li class="page-item active" aria-current="page"><span class="page-link">' + (i+1) + '</span></li>';
        }
        else {
          nav_html += '<li class="page-item"><a class="page-link" href="#">' + (i+1) + '</a></li>';
        }
      }
      $('#' + pTable.id).siblings('nav').children('ul').html(nav_html)
    }

    // 翻页按钮点击
    pTable.gotopage = function(pageNo) {
      pTable.showPage(pageNo);
      pTable.showPagination(pageNo);
      pTable.bindPagination();
    }

    // 翻页操作
    pTable.bindPagination = function() {
      $('#' + pTable.id).siblings('nav').find('a').click(function(){
        pageNo = parseInt($(this).text()) - 1;
        pTable.gotopage(pageNo);
      });
    }

    // 搜索按钮
    pTable.bindSearch = function() {
      $('#button-query').click(function(){
        patientName = $('#button-query').parent().siblings('input').val()
        pTable.search(patientName);
        pTable.gotopage(0);
      })
    }

    // 表格初始化
    pTable.init = function() {
      pTable.loadData();
      pTable.bindSearch();
    }

    return pTable;
  }
}
