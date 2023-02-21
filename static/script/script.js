$(function() {
    var rangePercent = $('[type="range"]').val();

    $('[type="range"]').on('change input', function() {
        rangePercent = $('[type="range"]').val();

        const Http = new XMLHttpRequest();
        Http.open('POST', '/')
        Http.send(rangePercent)

        $('h4').html(rangePercent + '<span></span>');
        $('[type="range"], h4>span').css('filter', 'hue-rotate(-' + rangePercent + 'deg)');
        // $('h4').css({'transform': 'translateX(calc(-50% - 20px)) scale(' + (1+(rangePercent/100)) + ')', 'left': rangePercent+'%'});
        $('h4').css({'transform': 'translateX(-50%) scale(' + (1+(rangePercent/100)) + ')', 'left': rangePercent+'%'});
    });
});



var chart;

        function requestData()
        {
            // Ajax call to get the Data from Flask
            var requests = $.get('/data');


            var tm = requests.done(function (result)
            {
                // var series = chart.series[0],
                //     shift = series.data.length > 10;

                // add the point
                chart.series[0].addPoint(result[0], true, chart.series[0].data.length > 5);
                chart.series[1].addPoint(result[1], true, chart.series[1].data.length > 5);
                chart.series[2].addPoint(result[2], true, chart.series[2].data.length > 5);
                chart.series[3].addPoint(result[3], true, chart.series[3].data.length > 5);
                

                // call it again after one second
                setTimeout(requestData, 2000);
                chart.redraw()
            });
        }

        $(document).ready(function() {
            chart = new Highcharts.Chart({
                chart: {
                    renderTo: 'data-container',
                    defaultSeriesType: 'spline',
                    events: {
                        load: requestData
                    }
                },
                legend: {
                    enabled: true
                },
                title: {
                    text: 'Live random data'
                },
                xAxis: {
                    type: 'datetime',
                    tickPixelInterval: 150,
                    maxZoom: 20 * 1000
                },
                yAxis: {
                    minPadding: 0.2,
                    maxPadding: 0.2,
                    title: {
                        text: 'Value',
                        margin: 80
                    }
                },
                series: [{
                    name: 'Random data 0',
                    data: []
                },
                {
                    name: 'Random data 1',
                    data: []
                },
                {
                    name: 'Random data 2',
                    data: []
                },
                {
                    name: 'Random data 3',
                    data: []
                }
                ]
            });

        });