
const allList = document.getElementById("allList");
const single = document.getElementById("single");
const ques = document.getElementById("ques");
const ans = document.getElementById("ans");
const prev = document.getElementById("prev");
const next = document.getElementById("next");
const root = window.location.origin + window.location.pathname.replace(/\/+$/, '');
const urlParams = new URLSearchParams(window.location.search);
var allData,
  startX = 0,
  endX = 0;


const banner = document.getElementById('banner');


ans.addEventListener('scroll', () => {
  if (ans.scrollTop > 20) {
    // Hide banner and topbar
    banner.classList.add('hidden');
    topbar.classList.add('hidden');
  } else {
    // Show banner and topbar
    banner.classList.remove('hidden');
    topbar.classList.remove('hidden');
  }
});



document.getElementById("hLink").setAttribute("href", root);
document
  .getElementById("rLink")
  .setAttribute("href", `${root}/?file=random`);

// Event listners for left/right navigation keys
document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight") {
    next.click();
  } else if (event.key === "ArrowLeft") {
    prev.click();
  }
});

// Event listners for left/right swiping
document.addEventListener("touchstart", (event) => {
  startX = event.touches[0].clientX;
});

document.addEventListener("touchend", (event) => {
  endX = event.changedTouches[0].clientX;
  if (startX < endX - 150) { //<- swipe distance
    prev.click();
  } else if (startX > endX + 150) {
    next.click();
  }
});

function fetchCall(url) {
  return fetch(url).then((response) => {
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
    }
    return response.json();
  });
}

async function fetchData(fileName) {
  let jsonFilePath;

  if (fileName == "random") {
    try { // fetch full list and randomly choose a single one
      let data = await fetchCall(`${root}/jsons/full.json`);
      let randomIndex = Math.floor(Math.random() * data.length); // Generate a random index
      jsonFilePath = `${root}/jsons/${data[randomIndex].file}.json`;
    } catch (error) {
      console.error("Error:", error.message);
      return;
    }
  } else {
    jsonFilePath = `${root}/jsons/${fileName}.json`;
  }
  // Now fetch a particular question-answer
  try {
    let info = await fetchCall(jsonFilePath);
    ques.innerHTML = info["question"].replace(/\\\\/g, "\\");
    ans.innerHTML = info["answer"].replace(/\\\\/g, "\\");
    // Latex syntax are modified to double slash during conversion 
    // also json can not store with single slach
    // but mathjax can not render those 
    // so they are modified back to single slash
    prev.setAttribute("href", `${root}/?file=${info["prev"]}`);
    next.setAttribute("href", `${root}/?file=${info["next"]}`);
    MathJax.typesetPromise([ques, ans]).catch((err) =>
      console.error("MathJax rendering failed: ", err)
    );
  } catch (error) {
    console.error("Error:", error.message);
  }
}

const fileName = urlParams.get("file");

if (fileName) { // render a single question-answer
  allList.classList.add("hide");
  single.classList.remove("hide");
  fetchData(fileName);
} else { // render the full question list
  allList.classList.remove("hide");
  single.classList.add("hide");

  fetchCall(`${root}/jsons/full.json`).then((data) => {
    document.getElementById("qlist").innerHTML = data
      .map(
        (elem) =>
          `<li><a href="${root}/?file=${elem.file}">${elem.question}</a></li>`
      )
      .join("\n");
      next.setAttribute("href", `${root}/?file=${data[0]["file"]}`);
  });
}


