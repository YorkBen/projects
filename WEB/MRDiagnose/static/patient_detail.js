var PatientObj = {
  createNew: function(pid) {
    var pObj = {}
    pObj.id = pid
    pObj.mr_no = 0;

    // 生成病历不同类型内容的展示代码
    pObj.addMRRecord = function(no, title, text) {
      head_id = "heading" + no;
      collapse_id = "collapse" + no;
      show = "";
      if (no == 0) {
        show = " show";
      }

      html = '<div class="card">';
      html += '<div class="card-header" id="' + head_id + '">';
      html += '<h2 class="mb-0">';
      html += '<button class="btn text-left font-weight-normal" type="button" data-toggle="collapse" data-target="#'
                + collapse_id + '" aria-expanded="true" aria-controls="' + collapse_id + '">';
      html += title;
      html += '</button></h2></div>';
      html += '<div id="' + collapse_id + '" class="collapse' + show + '" aria-labelledby="' + head_id + '" data-parent="#accordionMR">';
      html += '<div class="card-body">';
      html += text;
      html += '</div></div></div>';
      $('#accordionMR').append(html);
    }

    pObj.assembleMRText = function(mrobj, level) {
      html = '';
      if (typeof(mrobj) == 'string' || typeof(mrobj) == 'number') {
        html = mrobj;
      }
      else if (mrobj instanceof Array){
        for(var k=0; k<mrobj.length; k++) {
          html += '<p class="lv-' + level + '">' + pObj.assembleMRText(mrobj[k], level+1) + '</p>';
        }
      }
      else if (mrobj != undefined){
        let keys = mrobj['keys'];
        for(var i=0; i<keys.length; i++) {
          key = keys[i];
          html += '<p class="lv-' + level + '"><span style="font-weight: bold;">' + key + ':&nbsp;&nbsp;</span>' + pObj.assembleMRText(mrobj[key], level+1) + '</p>';
        }
      }

      return html;
    }

    pObj.processMROneType = function(key, mrdata) {
      text_html = '';
      if (mrdata !== undefined) {
        console.log(mrdata);
        text_html = pObj.assembleMRText(mrdata, 0)
      }
      pObj.addMRRecord(pObj.mr_no, key, text_html);
      pObj.mr_no += 1;
    }


    pObj.processMRText = function(mrdata) {
      console.log(mrdata)
      pObj.processMROneType('门诊病历', mrdata['门诊病历']);
      pObj.processMROneType('入院记录', mrdata['入院记录']);
      pObj.processMROneType('病案首页', mrdata['病案首页']);

      // 检查
      insp_data = {};
      keys = [];
      if ('超声' in mrdata) {
        insp_data['超声'] = mrdata['超声'];
        keys.push('超声');
      }
      if ('放射' in mrdata) {
        insp_data['放射'] = mrdata['放射'];
        keys.push('放射');
      }
      if ('病理' in mrdata) {
        insp_data['病理'] = mrdata['病理'];
        keys.push('病理');
      }
      insp_data['keys'] = keys;
      pObj.processMROneType('检查', insp_data);

      pObj.processMROneType('检验', mrdata['检验']);

      // 病程记录
      diec_data = {};
      keys = [];
      if ('首次病程' in mrdata) {
        diec_data['首次病程'] = mrdata['首次病程'];
        keys.push('首次病程');
      }
      if ('日常病程' in mrdata) {
        diec_data['日常病程'] = mrdata['日常病程'];
        keys.push('日常病程');
      }
      diec_data['keys'] = keys;
      pObj.processMROneType('病程记录', diec_data);

      pObj.processMROneType('出院记录', mrdata['出院记录']);
    }

    // 加载数据，第一次进入页面时请求
    pObj.loadMRData = function() {
      $.ajax({
    		url:"/patient_info/" + pObj.id,
        headers: {
          Accept: "application/json; charset=utf-8"
        },
    		data:{action:"jQueryAjax"},
    		type:"GET",
    		success:function (data) {
          if(data.status_code == 200) {
            pObj.processMRText(data.text)
          }
          else {
            alert(data['reason']);
          }
    		},
    		//返回数据的格式
    		dataType : "json"
    	});
    }

    // 表格初始化
    pObj.init = function() {
      pObj.loadMRData();
      // pObj.bindSearch();
    }

    return pObj;
  }
}
