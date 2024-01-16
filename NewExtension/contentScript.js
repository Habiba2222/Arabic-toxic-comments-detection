let twitterArray = [];
let tweetsPred = new Map();
let username = "";
let tweetscnt = 0;
let curUrl = "";
let prevUrl = "";
let curURL2 = "";
let prevUrl2 = "";
let result = "";
let videoId = [];

let date = /[a-z]*\/s+[0 - 9][0 - 9]/g;
/////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
// this function check if we are at the same page or not , if not => we clear all variables and to be set
// again for the new page
function getUserName() {
  username = "";
  curURL2 = window.location.href;
  // console.log("HEREEE");
  if (curURL2 != prevUrl2) {
    key = curURL2.split("https://twitter.com/");
    prevUrl2 = curURL2;
    //prevUrl2.replace(/(\?.*)/, "");
    // console.log((key[1].split("/").length));
    if (
      key[1] != "home" &&
      key[1] != "explore" &&
      key[1] != "messages" &&
      key[1] != "settings" &&
      key[1] != "notifications" &&
      key[1].split("/").length == 1
    ) {
      if (key[1].split("?").length != 1) {
        if (key[1].split("?")[0] != "search") {
          username = key[1].split("?")[0];
          // console.log(key[1].split("?")[0]);
        }
      } else {
        username = key[1];
      }
    }
  }
  return username;
}
///////////////////////////////////////////////////////
function adjust(resp) {
  if (resp == "Religion") {
    s = "religious";
  } else if (resp == "Race") {
    s = "racial";
  } else if (resp == "Ideaology") {
    s = "ideological";
  } else if (resp == "Social Class") {
    s = "classist";
  } else if (resp == "OFF") {
    s = "offensive";
  } else if (resp == "Disability") {
    s = "insensitive";
  }
  return s;
}
//////////////////////////////////////////////////////
const fakeDetect = async (username) => {
  let f = "";
  await fetch("http://localhost:5000/detect/fake", {
    method: "post",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: username }),
  })
    .then((response) => response.json())
    .then((data) => {
      data = data.fake;
      if (data == "0") {
        f = "LEGIT!";
        // console.log("LEGIT!");
      } else {
        f = "FAKE!";
        // console.log("FAKE!");
      }
    })
    .catch((err) => {
      console.log("nooooooooooooooooooooooooooooooo");
    });
  return f;
};
////////////////////////////////////////////////////////////////////////////////////
async function callDetect() {
  username = getUserName();
  if (username != "") {
    // console.log(username);
    result = await fakeDetect(username);
  }
  if (result == "FAKE!") {
    $('[data-testid="UserName"]').append(
      `<span style =" display: inline-block;
      padding: 10px 15px;
      font-size: 15px;
      text-align: center;
      text-decoration: none;
      outline: none;
      color: #fff;
      background-color:red;
      border: none;
      border-radius: 10px;
      margin-bottom:10px;
      margin: 10px;"> 
      WARNING: This might be a <span style="font-weight:bold">Fake Account</span>! </span>`
    );
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
async function CurrentURL() {
  curUrl = window.location.href;
  if (curUrl != prevUrl) {
    await callDetect();
    twitterArray = [];
    prevUrl = curUrl;
    cntTweets = 0;
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////

function cleanTweet(tweet) {
  tweet = tweet.replace(/[a-zA-Z]*/, "");
  //console.log(tweet);

  let cleanedtweet = tweet.replace(/^[^·]*/g, "");
  cleanedtweet = cleanedtweet.substring(1);
  let cleanedtweet1 = cleanedtweet.replace(/^[1-9]{1,2}/, "");

  if (cleanedtweet != cleanedtweet1) {
    cleanedtweet = cleanedtweet1.substring(1);
  }

  if (cleanedtweet.includes("Follow Us HereDownload"))
    cleanedtweet = cleanedtweet.split("Follow Us HereDownload")[0];
  //cleanedtweet = cleanedtweet.split("#")[0];
  // console.log(cleanedtweet);
  return cleanedtweet;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
async function modifyDOM() {
  let changed = false;

  ///////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////fetch user data//////////////////////
  // const user = async (username) => {
  //   let f = "";
  //   await fetch("https://api.twitter.com/1.1/users/show.json?screen_name="+username, {
  //     method: "GET",
  //     headers: { "authorization" : "Bearer AAAAAAAAAAAAAAAAAAAAAO6UdgEAAAAAUWyMfm6a0HdTmdYM%2BHaL4l21zU0%3DWU40sg62MNpwBCY376ftgERGwfWI2q3jIPc4XJJIVG8Cb2cJ7S" },
  //     // body: JSON.stringify({ url: videoid }),
  //   })
  //     .then((response) => response.json())
  //     .then((user_data) => {
  //       user_data = user_data.output;
  //       console.log(user_data);
  //       f = user_data;
  //     })
  //     .catch((err) => {
  //       console.log("the server is not working in video turn it on ");
  //     });
  //   return f;
  // };
  ///////////////////////////////////////////////////
  // if ($("article[role=article]").length) {
  //   $("article[role=article][mark!=2]").each(async function () {

  //   });
  // }

  ///////////////////////////////////////////////////////////////////////////////////////////

  // print tweets if a new tweet is detected

  ////////////////////////////////////////////////////////////////////////////////////////////

  // if (changed) {
  //  console.log(JSON.stringify(twitterArray));

  /////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////
  const aa = async (bodyyy) => {
    let f = "";
    let output = "";
    console.log(bodyyy);
    await fetch("http://localhost:5000/prediction/txt", {
      method: "post",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bodyyy),
    })
      .then((response) => response.json())
      .then((data) => {
        video = data.videoOutput;
        tweet = data.tweetOutput;
        if (bodyyy.is_video == 0) {
          output = tweet;
        } else if (video == "NOT_HS" && tweet == "NOT_HS") {
          output = "NOT_HS";
        } else if (video != "NOT_HS" && tweet == "NOT_HS") {
          output = video;
        } else if (video == "NOT_HS" && tweet != "NOT_HS") {
          output = tweet;
        } else {
          if (video == tweet) {
            output = tweet;
          } else {
            output = tweet + " and " + video;
          }
        }
        // console.log(output);
        f = output;
      })
      .catch((err) => {
        console.log("the server is not working turn it on ");
      });
    return f;
  };

  //}

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////

  $('[data-testid="tweet"]').each(async function (index) {
    // var tweetContainer = $(this);
    // hasVideo = tweetContainer.find("video");
    // console.log("try to detect video element.... " + hasVideo.length);
    // if (hasVideo.length==0)
    // {
    let video_id = 0;
    let is_video = false;
    let tweet = $(this).text();
    // console.log("TWEETT TEXT BEFORE" + tweet);
    let finaltweet = cleanTweet(tweet);
    // console.log("TWEETT TEXT AFTER" + finaltweet);

    /////////////////////////////////////////////////////////////////////////
    var tweetContainer = $(this);

    hasVideo = tweetContainer.find("video");
    // console.log("try to detect video element.... " + hasVideo.length);
    if (hasVideo.length) {
      // console.log("[3] video element found!");

      var fullLinkToTweet = document.location.href;

      if (fullLinkToTweet.indexOf("/status/") === -1) {
        var linkFromTweetsList = tweetContainer.find("a.r-3s2u2q[role=link]");
        fullLinkToTweet =
          "https://twitter.com" + linkFromTweetsList.attr("href");
      }

      video_id = fullLinkToTweet.split("status/")[1];
      is_video = true;

      tweetContainer.attr("mark", 2);
    }

    ////////////////////////////////////////////////////////////////////////////
    let body = { video_id: video_id, is_video: is_video };

    // console.log(body);

    if (
      (!twitterArray.includes(finaltweet) && finaltweet != "") ||
      (!videoId.includes(video_id) && is_video)
    ) {
      videoId.push(video_id);
      twitterArray.push(finaltweet);
      body = Object.assign(body, { predict: finaltweet });
      // console.log(body);
      // let resp = await aa(finaltweet);
      let resp = await aa(body);
      let output = "";
      let s1 = "";
      let s2 = "";
      if (resp != "NOT_HS" && resp != "") {
        if (resp.split(" and ").length == 1) {
          output = adjust(resp);
        } else {
          first = resp.split(" and ")[0];
          second = resp.split(" and ")[1];
          s1 = adjust(first);
          s2 = adjust(second);
          output = s1 + " and " + s2;
        }

        // else if (resp == "Gender")
        // {

        // }
        $(this).addClass("squished");
        $(this).html(
          `<button class = "tweet" style =" display: inline-block;
                      padding: 10px 15px;
                      font-size: 15px;
                      cursor: pointer;
                      text-align: center;
                      text-decoration: none;
                      outline: none;
                      color: #fff;
                      background-color:#17A8F4;
                      border: none;
                      border-radius: 10px;
                      margin-bottom:10px;
                      margin: 10px;"
                      data-original-content="${encodeURI($(this).html())}"
                      mark="1"> This tweet contains ${output} content. </button>`
        );
        await chrome.runtime.sendMessage(
          { message: "listeners" },

          function (response) {
            //  console.log(response);
          }
        );
      }
    }
    // }
  });
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
async function detectChange() {
  // var work = window.localStorage.getItem("work");
  // if (work != undefined && work == 1) {

  await CurrentURL();
  modifyDOM();
  //}
}

setInterval(detectChange, 3000);
