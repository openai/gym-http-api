"use strict";

var gulp = require("gulp"),
    shell = require("gulp-shell"),
    path = require("path"),
    tsc = require("gulp-typescript");

var tsOptions = {
    module: "commonjs",
    target: "es5",
    removeComments: true,
    preserveConstEnums: true,
    sourceMap: true,
    jsx: "react"
};

var srcDirectory = path.join(__dirname, "src"),
    distDirectory = path.join(__dirname, "dist");

gulp.task("build_examples", function () {
    var examplesSrc = path.join(srcDirectory, "examples", "*.ts"),
        examplesDist = path.join(distDirectory, "examples");
    var tsResult = gulp.src(examplesSrc)
        .pipe(tsc(tsOptions));
    tsResult.dts.pipe(gulp.dest(examplesDist));
    return tsResult.js
        .pipe(gulp.dest(examplesDist));
})

gulp.task("build_library", function () {
    var librarySrc = path.join(srcDirectory, "lib", "*.ts"),
        libraryDist = path.join(distDirectory, "lib");
    var tsResult = gulp.src(librarySrc)
        .pipe(tsc(tsOptions));
    tsResult.dts.pipe(gulp.dest(libraryDist));
    return tsResult.js
        .pipe(gulp.dest(libraryDist));
});

gulp.task("typescript", ["build_examples", "build_library"]);
gulp.task("prettify", shell.task("./ts_format.sh"));
gulp.task("build", ["prettify", "typescript"]);
gulp.task("default", ["build"]);